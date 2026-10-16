# 单卡共享 + Top-1 MoE 原型

实现提案：`openspec/changes/single-gpu-shared-expert-moe/`。结构、CUDA 执行、训练和导出入口已实现；**尚未完成正式训练**。随机包用于数值/执行/速度验证，少步教师蒸馏包只验证训练管线，不能据此判断纹理质量。三种主尺寸的随机原型速度门禁已通过，实际阶段结果见同目录的 [SHARED_MOE_REPORT.md](SHARED_MOE_REPORT.md)。

## 结构与语义

`shapes/student_shared_moe_v1.json`：79,933,097 个可训练参数，公共参数 200,105（0.2503%）；另有 480 个 FP32 路由偏置 buffer。每个 bank 为 `[17, in, out]`：第 0 份是始终激活、可训练的共享专家，第 1～16 份是路由专家。对应 17 份矩阵/FFN 的形状和每 token 运算完全相同。总参数按所有 bank 计数，逐层两份激活参数等效量为 9,580,457；它不代表整图只访问两份固定专家，也不直接代表耗时。

每个空间块按不移位的 8×8 窗均值路由；ViT 按 token 路由。块内 FFN/QKV/输出投影共用块输入的路由。11 个过渡投影分别路由：编码投影在池化后，解码投影在上采样/skip 合并前，ViT 进出投影在 d5 网格。输入 16 lane、几何、attention 相位、head 和 composite 位置保持原样。

路由全过程 FP32，无 jitter，无容量配额、重新指派或丢弃：

```text
z = Linear16(LayerNorm(window_mean_or_token(x)))
e = argmax(z + frozen_bias)       # 同分取最小编号
g = 16 * softmax(z)[e]            # gate 不含 bias；不把 Top-1 重新归一为 1
FFN = (F_shared(x) + g * F_e(x)) / 2
u = x * aux_ffn + FFN
qkv = (QKV_shared(u) + g * QKV_e(u)) / 2
a = Attention(qkv)               # 只执行一次，仍是原来的 attention
y = u + aux_attn * (P_shared(a) + g * P_e(a)) / 2
```

每个 FFN 各自先激活再融合。此语义不是两套完整 attention 网络结果的平均。过渡矩阵的 bias 也分成 17 份，参与相同的两路融合。

`nr_shared_moe.py` 为 Python 参考；`shared_moe_cuda.cuh`/`engine_cuda.cu` 为原生执行。共享分支直接使用全行矩阵乘；路由分支稳定紧凑打包，由设备端 offsets/tile 描述调度变长 FP32 GEMM。不构造 `16 * 最大组行数` 的容量张量。小网格用 BlockScan，大网格用 CUB 稳定 radix sort/scan；CUDA Graph 内更新描述，重放不回读负载或分配 scratch。C=H=32 的 FFN 融合两次 GEMM 与中间 SiLU，减少临时矩阵读写。

## 构建与环境

Windows 使用兼容 CUDA Toolkit 的 MSVC 开发者终端：

```powershell
cmake -S engine -B engine/build-cuda -DNR_CUDA=ON
cmake --build engine/build-cuda --config Release --target gpu_engine cpu_engine shared_moe_check
```

NMake 可执行文件位于 `engine/build-cuda/`；Visual Studio 生成器通常位于 `engine/build-cuda/Release/`，下列 `--engine` 据此调整。Linux 使用相同 CMake 目标，文件没有 `.exe` 后缀。只运行原生推理不依赖 Python。

校验脚本使用项目已有 CUDA Python 环境即可。本次环境是 `tmp/teacher-cuda-venv`（Torch 2.6.0+cu124、Python 3.10）。如需新环境，用 uv：

```powershell
uv venv .venv-moe --python 3.10
uv pip install --python .venv-moe/Scripts/python.exe torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv-moe/Scripts/python.exe numpy==2.2.6 pillow==12.3.0
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '8'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
```

Linux 的 Python 路径为 `.venv-moe/bin/python`；用 `export` 设置相同变量。命令中的 `python` 指上述 CUDA 环境。

## 审计、原型与数值检查

```powershell
python teacher/test_shared_moe.py
python teacher/test_cost_model.py
python teacher/cost_model.py --shape shapes/student_shared_moe_v1.json --sizes 320,512 --json
python teacher/prototype_shared_moe.py --preset compact --output tmp/shared_moe/compact
python teacher/audit_moe.py tmp/shared_moe/compact --verify --output tmp/shared_moe/compact/audit.json
python teacher/check_shared_moe.py --weights tmp/shared_moe/compact --output tmp/shared_moe/parity
engine/build-cuda/shared_moe_check.exe
```

`prototype_shared_moe.py` 保存种子、候选参数/成本、逐张量审计、实际负载、门控、分支 RMS 和包往返逐位一致记录。原型包标明 `untrainedPrototype=true`，有独立版本化 routing/precision/layout、shape SHA256、冻结零偏置来源和分片 SHA256。公共参数和 buffer 分开计数；审计拒绝缺失过渡或不匹配 bank。旧包继续走原来的 Python/C++ 分支；CPU 加载此新包明确报错。

跨实现检查显式使用 raw head `atol=3e-5, rtol=2e-4`，并报告 max/mean/p99/RMS；合成 RGB 使用 head/4。该阈值覆盖大图 FP32 不同累加树的微小误差。四个小尺寸通常只有几 e-6；原图选择单独精确比较，不允许路由差异被数值容差掩盖。大图可加 `--routes-only` 只保留路由 capture，仍检查 eager/Graph 和完整 head。

`check_shared_moe.py` 共用 features，逐层精确比较选择，数值比较 gate/FFN/QKV/attention/过渡/head，检查 eager/Graph/重复执行。默认尺寸包括 320²、512²、320×512、137×201。原图检查可单独运行，输出全部放在 `tmp`，不修改样例：

```powershell
python teacher/check_shared_moe.py --weights tmp/shared_moe/compact --head-only --output tmp/shared_moe/native --images teacher/samples/t1/t1.png teacher/samples/t2/t2.png teacher/samples/t3/t3.png teacher/samples/t4/t4.png teacher/samples/t5/t5.png teacher/samples/t6/t6.png teacher/samples/t7/t7.jpg
compute-sanitizer --tool memcheck --error-exitcode 99 engine/build-cuda/shared_moe_check.exe --quick
compute-sanitizer --tool racecheck --error-exitcode 99 engine/build-cuda/shared_moe_check.exe --quick
```

原图模式先用既有原生图像入口产生默认条件的 features，再在两端读取同一份 features；报告 head 误差与合成后 uint8 差异。`--head-only` 不转储大型中间量，Graph/eager 一致性由小尺寸全量检查覆盖。单独的压力工具在**同一个 Graph** 重放均匀、全热、90% 热点、空组、同分/近同分和非零偏置，验证紧凑顺序、完整行、固定 FMA 顺序及重复确定性，并输出路由+打包+GEMM 微基准。

## 完整前向门禁与诊断

```powershell
python teacher/bench_shared_moe.py --candidate tmp/shared_moe/compact --output tmp/shared_moe/gate --engine engine/build-cuda/gpu_engine.exe --reference-python tmp/teacher-cuda-venv/Scripts/python.exe
```

固定同卡、FP32、TF32 off、同 features、实际 field；每轮 10 次预热、100 次采样，三个主尺寸各做三轮 baseline/candidate 交替。CUDA event 覆盖 router、分组、两路矩阵、一次 attention、所有过渡和融合。初始化、Graph 构建、H2D/D2H、图像/lane/composite/PNG 不计入前向。每尺寸轮间 median 比值中位数 ≤1.00，p95 轮间中位数比值 ≤1.10 才通过；接近但大于 1 也不算通过。

报告保存每次耗时、环境、包/shape/源文件/可执行文件/features hash、提交、条件、field 和 buffer 分配高水位。运行时不要并行执行其他 GPU 校验或训练。峰值显存诊断与计时分开：设置 `NR_CUDA_MEMORY=1` 单独执行一次，会在设备分配/算子边界采样设备级显存，包含其他应用；同时报告本进程自有 buffer 的精确分配高水位（不含 CUDA/cuBLAS/Graph 内部分配）。采样值有明确边界，不冒充独占进程完整显存峰值。

```powershell
python teacher/profile_shared_moe.py --weights tmp/shared_moe/compact --features tmp/shared_moe/gate/320x320_features.bin --output tmp/shared_moe/profile
$env:NR_CUDA_MEMORY = '1'
engine/build-cuda/gpu_engine.exe --weights tmp/shared_moe/compact --features tmp/shared_moe/gate/320x320_features.bin --width 320 --height 320 --bench --warmup 1 --repeats 1
Remove-Item Env:NR_CUDA_MEMORY
```

`profile_shared_moe.py` 使用带外部 event 节点的独立 Graph 收集分段诊断，额外 event 有开销；block/route/transition 时间包含其子项，不能重复累加或作为正式门禁。成本模型输出理想实际行 MAC、一次 padded reference attention、非 MAC 工作计数与根据实测 load 计算的 tile 对齐浪费；没有实测 load 时对齐项为 unknown，不假设均衡，也不套旧教师的吞吐系数预测新结构耗时。

下面的检查直接读取 benchmark 的同一份 features，输出逐层实际负载、理想 MAC，以及路由 GEMM 的行/列 tile 对齐额外 MAC。对齐项是当前内核算术计账，不是延迟预测：

```powershell
python teacher/check_shared_moe.py --weights tmp/shared_moe/compact --routes-only --features-root tmp/shared_moe/gate --sizes 320x320,512x512,1280x720 --output tmp/shared_moe/benchmark_routes
```

## 训练、恢复与导出

`train_distill.py` 检测新 kind 后默认 `student_precision=fp32`、`router_jitter=0`、`moe_aux=0`、`moe_z=0`，关闭 TF32。显式不兼容的 BF16/jitter/QAT 配置报错。可选 z-loss 使用 `mean(logsumexp(z)^2)`；旧网络保留原 BF16/抖动/L2 复现路径。

训练控制器仅统计主任务前向的实际 field 行，跨微批累积后在 optimizer step **之后一次**更新：`b = center(b + rate*sign(sum(load)/16 - load))`。默认 `moe_bias_rate=0.001`。共享专家和 GEMM 对齐空行不计负载；时序 teacher-forcing 的 no_grad 次前向也不重复计入。评估和推理不调用控制器。

checkpoint 除 model/EMA/optimizer 外保存 controller 版本/偏置累积/更新位置/随机状态。正在累积时的 helper 还保存梯度和已消费微批计数，恢复可继续同一步；常规 CLI 在完整 step 边界保存。EMA 对可训练权重做指数平均，对路由偏置在 controller 更新后**复制快照**，避免把控制状态做平均而生成不存在的边界状态。`routing.jsonl` 包括每层负载、最后激活 step、激活步数、门控/分数间隔/偏置变化、共享/路由幅值和逐专家梯度；router 的任务/正则梯度贡献分开记录。

示例训练配置应明确保存教师来源与条件；本次只运行 320/512 的短 smoke，没有启动长训：

```json
{
  "shape": "shapes/student_shared_moe_v1.json",
  "teacher_weights": "weights/teacher",
  "size": "320,512",
  "size_weights": "0.7,0.3",
  "batch": 2,
  "student_precision": "fp32",
  "router_jitter": 0,
  "moe_bias_rate": 0.001,
  "moe_aux": 0,
  "moe_z": 0,
  "comet_disabled": true
}
```

```powershell
python teacher/test_moe_training.py
python teacher/convert_weights.py --shape student runs/example/ckpt.pt --student-state ema -o weights/example_ema
python teacher/convert_weights.py --shape student runs/example/ckpt.pt --student-state model -o weights/example_raw
```

EMA 仍是旧 CLI 的默认来源。新包要求完整 optimizer/controller 边界、非歧义的 EMA bias 来源，记录 checkpoint hash、教师/学生提交和源代码 hash、教师 manifest hash、步数、冻结偏置、公共参数清单及全部推理语义；不静默把旧 8 专家包改成新结构。阶段报告中的少步 smoke 导出不能视为完成训练。

质量诊断入口支持真实教师固定版本、原图尺寸、两组确定噪声、用户给定皮肤 ROI、残差高频增益/相关性、共享/路由分支移除以及独立历史反馈链。ROI 必须明确给定；不会把整图 PSNR 或均衡负载自动判为纹理学会：

```powershell
python teacher/eval_shared_moe.py --weights weights/example_ema --teacher-weights weights/teacher --images frames/0.png frames/1.png --temporal --roi 100,80,400,420 --output tmp/shared_moe/quality.json
```

默认不缩放图片。只有显式 `--size` 才改变评测尺寸；随机包必须使用 `--smoke` 并输出 `pipeline-smoke-only`。Windows 若尚未安装教师的原生库，可用 `NR_CUDA_LIBRARY` 指向已编译的 `nr_kernels` DLL；此选项仅影响教师诊断入口。
