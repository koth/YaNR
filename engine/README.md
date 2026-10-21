# 学生模型 C++ 推理

`cpu_engine` 可以直接加载 `weights/student_mixed` 发布包、读取图片、构建 16 lane 输入、
执行前向并输出 NR PNG。运行链路不需要 Python、PyTorch 或 OpenCV。
保留原来的 `--idx/--features/--out` 接口，供逐层对账使用。
同一个入口也支持 `weights/student_moe_v0` 的窗口/Token top-1 MoE。

## 批量重跑 t1～t7

脚本默认加载 `weights/student_v43`，使用 `style=2、tone=1、structure=1、skin=-1、automask=1`，
seed 为 12345。保持原图尺寸，首帧历史输入为当前原图，直接覆盖
`teacher/samples/t1`～`t7` 各目录的 `nr.png`，不创建备份。
从任意工作目录运行均可，默认权重和图片路径相对于仓库。

Windows（先按下文构建引擎；优先查找 CUDA 版本，否则使用 CPU）：

```powershell
.\engine\run_samples.ps1
# 仅跑指定图片，或指定其他权重/引擎
.\engine\run_samples.ps1 -Samples t1,t7 -Weights weights/student_v43
.\engine\run_samples.ps1 -Engine engine/build/Release/cpu_engine.exe
```

CUDA runtime DLL 需在 PATH 中；设置了 `CUDA_PATH` 时，Windows 脚本会自动加入其 `bin`。

Linux（在仓库根目录配置；图像解码只依赖 Pillow，推理无需 PyTorch）：

```bash
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build --target cpu_engine cpu_lanes -j 8
uv venv tmp/samples-venv --python 3.10
uv pip install --python tmp/samples-venv/bin/python pillow
bash engine/run_samples.sh
# 仅跑指定图片；有 CUDA 构建时可指定 GPU 引擎
bash engine/run_samples.sh t1 t7
# CUDA 构建还需生成 cpu_lanes，或用 --lanes 指定它
bash engine/run_samples.sh --engine engine/build-cuda/gpu_engine --lanes engine/build/cpu_lanes
```

Linux 脚本优先查找 `engine/build-cuda/gpu_engine`，其次是 `engine/build/cpu_engine`；
默认使用引擎同目录的 `cpu_lanes`。因为 Linux 图片入口尚不支持解码，
`tools/decode_proxy.py` 只负责读取 PNG/JPG 等图片并写入临时 RGB f32 数据；
输入 lanes、网络前向、合成和 PNG 写出均由 C++ 完成，临时数据运行后清理。
`--python` 可指定现有的 Pillow 环境；`--help` 查看路径和控制参数选项。
Windows 使用 `-Style/-Tone/-Structure/-Skin/-NoAutomask/-Seed/-Threads/-Device`，
Linux 对应 `--style/--tone/--structure/--skin/--no-automask/--seed/--threads/--device`。
任一图片或引擎运行失败时立即停止并返回非零状态。

## 构建与运行（Windows）

在仓库根目录执行，需安装 Visual Studio C++ 工具链和 CMake：

```powershell
cmake -S engine -B engine/build -DNR_AVX2=ON -DNR_VNNI=ON
cmake --build engine/build --config Release -j 8

# 原尺寸推理，输出到指定位置
engine/build/Release/cpu_engine.exe photo.jpg --weights weights/student_moe_v0 -o out/photo_nr.png

# 显式指定旧 student_mixed 训练范围内的 style
engine/build/Release/cpu_engine.exe photo.jpg --weights weights/student_mixed --style 1.5 -o out/photo_style.png

# student_moe_v0：完整 MoE 权重包
engine/build/Release/cpu_engine.exe photo.jpg --weights weights/student_moe_v0 -o out/photo_moe.png

# 512x512 推理 + 只测前向（预热 5 次，测量 30 次）
$env:OMP_NUM_THREADS = '8'
engine/build/Release/cpu_engine.exe photo.jpg --weights weights/student_mixed --size 512 `
    --style 0 --tone 0.5 --structure 0.5 --skin -1 --no-automask `
    -o out/photo512_nr.png --bench --warmup 5 --repeats 30

# 非方形输出
engine/build/Release/cpu_engine.exe photo.jpg --weights weights/student_moe_v0 `
    --width 512 --height 320 -o out/photo_rect_nr.png
```

只传图片也可以：权重默认 `weights/student_mixed`（相对于当前目录），
输出默认是输入图片旁的 `<stem>_nr.png`。
`--size` 将图片缩放为正方形；`--width/--height` 指定各轴尺寸，未指定的轴保持原尺寸。
CPU/GPU 图片入口和 C++ `LaneParams` 默认均为：`style=64、tone=1、structure=1、skin=1、automask=开启`。
`--style` 是 style ID/条件值，名义范围为 `0..127`，默认 `64`，不是后处理强度旋钮。
`student_mixed` 是旧 checkpoint，训练覆盖的 style 范围是 `0..3`；较大的值会把模型推到
训练分布外，可能出现背景噪声、颜色溢出或结构伪影。需要使用大范围 style 时请改用
`weights/student_moe_v0`，它按 `0..127` 全量程训练。旧 `student_mixed` 可将 `--style` 显式设为 `0..3` 范围内的值。
`--tone` 和 `--structure` 会直接写入条件 lane；`--skin` 仅在自动 mask 开启时生效，
负值跟随 structure。`--no-automask` 关闭自动 mask，`--automask` 显式开启；
这是 Python 教师入口的既定语义。使用 `--features` 时输入 lane 已经预先生成，命令行条件参数不会重建它。
有效尺寸至少 33，各轴上限 8192，d0 网格必须满足模型的 8 倍数约束。
模型会自行生成 padding，PNG 只保留有效图片区域。

图片解码使用 Windows WIC，支持 PNG/JPEG/BMP 等系统解码格式和中文路径；
缩放使用 WIC Fant，与 Python 工具的 PIL Lanczos 缩放结果可能不同。
Linux 上仍可使用发布包 + `--features` 或原有 `.idx` 接口，图片入口目前仅支持 Windows。

## Windows GPU 加速（NVIDIA CUDA）

`NR_CUDA=ON` 添加原生 CUDA / cuBLAS 后端，支持 `ffn`、`expert`、`split`、`moe`
和 ViT / MoE ViT，可直接加载现有 `student_mixed`、`student_moe_v0` 权重包或 `.idx`。
不需要 Python、PyTorch、ONNX 或额外的权重导出。CUDA 为可选依赖，默认 CPU 构建不需要它。

安装 NVIDIA 驱动、CUDA Toolkit、Visual Studio C++ 工具链和 CMake（至少 3.18）。
在 **x64 Native Tools Command Prompt for VS 2022** 中进入仓库根目录执行：

```bat
cmake -S engine -B engine/build-cuda -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DNR_CUDA=ON
cmake --build engine/build-cuda --target gpu_engine cpu_engine
```

本机 CUDA 12.4 / MSVC 14.38 / RTX 3080 已验证。若使用同样的安装位置，可先在 CMD 执行
`"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.38`。
Toolkit 不在默认位置时，配置命令加上 `-DCUDAToolkit_ROOT=D:/CUDA/v12.4`
和 `-DCMAKE_CUDA_COMPILER=D:/CUDA/v12.4/bin/nvcc.exe`。
默认生成 75 / 86 架构；只为本机 RTX 3080 构建可加 `-DCMAKE_CUDA_ARCHITECTURES=86`。
安装了 CUDA 的 VS 集成组件时，也可以使用 Visual Studio 生成器和 `--config Release`；
这种情况下可执行文件在 `engine/build-cuda/Release/`。

运行时需让 CUDA Toolkit 的 `bin` 目录位于 `PATH`（cuBLAS / CUDA runtime DLL）：

```powershell
engine/build-cuda/gpu_engine.exe photo.jpg --weights weights/student_moe_v0 `
    --size 512 -o out/photo_gpu.png

engine/build-cuda/gpu_engine.exe teacher/samples/lake.png --weights weights/student_moe_v0 `
    --style 0 --tone 0.5 --structure 0.5 --skin -1 --no-automask `
    --size 512 --bench --warmup 10 --repeats 50 -o out/lake_gpu.png

# 同一入口显式选后端；选择第 0 张 CUDA 显卡
engine/build-cuda/cpu_engine.exe photo.jpg --backend cuda --device 0 `
    --weights weights/student_moe_v0 -o out/photo_gpu.png
```

`gpu_engine` 默认 CUDA，`cpu_engine` 默认 CPU；两个入口共用图片、条件参数和 PNG 管线。
没有编译 CUDA 时请求 `--backend cuda` 会报告构建提示。CUDA 初始化失败会报告错误。
当前是 NVIDIA CUDA 支持，AMD / Intel 显卡使用 CPU 后端。

GPU 前向采用 FP32 cuBLAS GEMM、GPU LayerNorm/窗口池化、稳定排序的配额平衡路由、
按专家分组的 batched GEMM，以及融合的余弦窗口/全局注意力和残差算子。
窗口的 query 掩码、边界零填充、旧 expert-major reshape 及 static-image composite 语义沿用 CPU。
TF32 关闭，保持路由和输出精度；`NR_INT8` / `NR_PROFILE` 是 CPU 专用，CUDA 入口遇到它们会提示移除。
当前窗口边长支持 1..8，ViT padding token 上限 8192，路由通道上限 1024；
大图片还需足够的显存。

权重、输入和 scratch 驻留显存；预热后用 CUDA Graph 回放，前向没有路由回传或动态分配。
`--bench` 用同步的 CUDA event 报告全网络前向 median/min/p90，包含路由与所有网络算子，
排除 H2D/D2H、图片、lane、composite、PNG、CUDA 初始化和 Graph 构建。
H2D/D2H 单独报告；整个图片处理的耗时还需加上这些步骤。
`--dump-dir` 切换到逐步执行并转储与 CPU 相同的中间量，诊断模式不用于测速。

`cnn_unet_fast` 的 CNN CUDA 路径已优化 GroupNorm、im2col 和像素 MoE。
2026-10-20 本机 RTX 3080 的 FP32 / TF32 off 前向：320² **14.73 ms**、
512²（field 576×512）**35.69 ms**，分别较优化前快 **3.32× / 3.90×**。
采样方法、内存占用与数值对账见 [CNN CUDA 性能记录](CNN_CUDA_PERF.md)。

数值校验使用 uv 创建的现有 `.venv`，仅需要 CPU 版 PyTorch 作参考：

```powershell
$env:OMP_NUM_THREADS = '8'
.venv/Scripts/python.exe teacher/check_gpu.py --build engine/build-cuda --cpu-build engine/build/Release
```

初版验证中，两种真实权重的 320×320、512×512、320×512 和 137×201 均通过逐层 CPU / CUDA
及 PyTorch head 对账，包含 style 0、1.5、3/127。最大 head 误差不超过 `1.8e-5`
（阈值 `2e-3`）；CUDA eager 与 Graph 回放、独立重复运行的 head 逐位一致。
CUDA Compute Sanitizer 的 MoE 320×320 内存检查与线程竞争检查均通过（0 errors / 0 hazards）。

2026-10-14 以 `student_v3/v4` 为验收对象。v3 保留控制参数、尺寸与 history 覆盖，
9/9 通过；v4 仅测试默认 `style=64, tone=1, structure=1, skin=1, automask=on`，
320×320、512×512 均通过，最大 head 误差 `1.240e-5`。
这 11 组的输入 lanes 逐位一致，静态 PNG 最大通道误差为 `1/255`。
两个回归脚本均限制 v4 只测默认参数；复现命令和已修复的 half 舍入问题见
[PARITY.md](PARITY.md)。

窗口注意力对常用 4×4/8×8 使用编译期尺寸，保留其他 1..8 窗口的通用路径。
K 在共享内存中转置并加一个 padding 元素；softmax 每个 key 只归一化一次，
随后广播给 value 通道，保持原来的逐项累加和 FP32 舍入。8×8 使用 512 个线程。
MoE 两次稳定排序分别只处理 expert ID + margin 的有效位、32 位 margin；
不改变专家选择、配额、相同 margin 的稳定顺序或全 1 的非 victim 哨兵。
对比优化前可追加 `--baseline-gpu <旧版gpu_engine.exe>`，检查所有 case 的完整
head 是否逐位一致；Windows 使用 `$env:PYTHONUTF8 = '1'` 读取 UTF-8 shape 文件。

C++ `infer_bgra` 提供同尺寸的时序输入/输出路径。先用 `upload` 初始化不变的
noise/control lanes；之后上传前后两帧 BGRA8（支持不同的行 stride），在 GPU
只更新 lanes 4..9，并转换 FP32 head 为 BGRA8 + confidence。256 项查表沿用
CPU 的三次 half 舍入，包含镜像并 clamp 的边界；输出 RGB 累加/量化保留原顺序。
输入和输出 staging 一次分配为 page-locked memory，稳态只在最终下载后等待一次。
`CudaBgraTimings` 的各项加和为 host elapsed：prepare 含 host copy/GPU lanes，
transfer 为 H2D+D2H device events，output 为 GPU 转换/host copy，forward 含
提交/等待开销；`device_forward_ms` 单列纯网络 event 时间。`download_features`
用于核对全部 lanes。CLI 仍使用原 float pipeline，`--bench` 继续只计纯网络。

### CUDA 前向 benchmark

2026-10-13 本机实测：RTX 3080 10GB、NVIDIA 驱动 566.36、CUDA 12.4、MSVC Release；
CPU 对照为 i9-10900K / AVX2 / 8 个 OpenMP 线程。输入 `teacher/samples/lake.png`，
style 0、tone/structure 0.5、seed 12345，两种后端均预热 10 次、测量 50 次。
Windows 桌面应用在运行，GPU 承担显示任务，后台负载会影响结果。

| 权重 | 有效尺寸 | 实际 field | CPU median | GPU median | GPU p90 | 前向加速 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `student_mixed` | 320×320 | 320×320 | 28.143 ms | 4.923 ms | 5.766 ms | 5.7× |
| `student_mixed` | 512×512 | 576×512 | 80.180 ms | 11.177 ms | 11.629 ms | 7.2× |
| `student_moe_v0` | 320×320 | 320×320 | 84.497 ms | 8.374 ms | 8.818 ms | 10.1× |
| `student_moe_v0` | 512×512 | 576×512 | 228.584 ms | 17.276 ms | 17.965 ms | 13.2× |

上表为窗口注意力优化前的初版 engine。优化后的 WOF 原始截图（384×224，
实际 field 384×320，seed 0）预热 10 次、测量 50 次：GPU median **6.317 ms**、
p90 **6.894 ms**；保存的旧版同图 median 8.525 ms。完整 FP32 head 的 SHA256
一致。MAME 游戏内纯网络约 7.074 ms，整段 BGRA adapter 约 7.530 ms（包含
两帧 staging、GPU lanes、H2D/D2H、输出转换和等待；排除最终 BGFX 上传）。
独立 benchmark 与同时进行 BGFX 渲染的游戏测量不能视为相同负载或硬时延保证。

MoE 的 320 / 512 H2D 输入约 1.31 / 3.29 ms，D2H head 约 0.34 / 0.75 ms，
均在上表前向时间之外。CUDA 上下文与权重上传也另计；反复启动进程会重复加载。
目前所有前向算子都是 FP32，以上结果没有采用 FP16 或 TF32 近似。

## 前向计时

`--bench` 仅对 `Engine::forward(features, head)` 计时，报告 median、min、p90。
权重读取/打包、图片解码/缩放、lane 构造、composite 和 PNG 写入均在计时之外。
预热也排除在统计之外。median 为中间两项的平均（偶数样本），p90 使用 nearest-rank。
不带 `--bench` 时打印的是第一次前向，包含 scratch/cache 初始化，不能当成稳定吞吐。

线程数默认沿用 `OMP_NUM_THREADS`，未设置时为 8；`--threads` 可以覆盖它。
测速时不设置 `NR_PROFILE`，它会增加内部计时开销。
`NR_INT8=1` 仍可启用既有混合 int8 路径，默认使用 fp32。
`--dump-dir` 和 `--bench` 不能同时使用，中间张量复制会污染前向测速。

2026-10-09 本机 FP32 复测：i9-10900K，MSVC Release /O2、AVX2、8 个 OpenMP 线程；
使用 `teacher/samples/lake.png` 和真实 `student_slim_mixed` step 59999 权重
（4,286,790 参数），预热 10 次、测量 50 次；后台负载/调度会影响结果。

| 有效尺寸 | 实际 field | median | p90 |
| --- | --- | --- | --- |
| 320×320 | 320×320 | 23.868 ms | 27.688 ms |
| 512×512 | 576×512 | 65.274 ms | 75.091 ms |

### `student_moe_v0` benchmark

这组数据使用 `weights/student_moe_v0`、MSVC Release /O2、AVX2、8 个 OpenMP
线程、10 次预热和 30 次前向测量；计时只包含 `Engine::forward`，权重加载、图片和 PNG
均在计时外。MoE 路由与 Python 包的 320/512 head 最大误差为 `5.7e-6/6.7e-6`。

| 有效尺寸 | 实际 field | FP32 median | FP32 p90 | mixed INT8 median | mixed INT8 p90 |
| --- | --- | ---: | ---: | ---: | ---: |
| 320×320 | 320×320 | 80.836 ms | 90.004 ms | 123.503 ms | 130.744 ms |
| 512×512 | 576×512 | 232.554 ms | 245.718 ms | 289.687 ms | 299.804 ms |

当前 i9-10900K 没有 VNNI，mixed INT8 使用 AVX2；本机上 MoE 的分组/量化开销使 INT8
暂时慢于 FP32，默认仍建议 FP32。VNNI 硬件可直接用同一命令和 `NR_INT8=1` 测量。

前面的 `student_mixed` 数字使用 10 次预热、50 次测量；MoE 表使用 10 次预热、30 次测量。
优化点是：
普通 FFN 和 expert FFN 的 expand/SiLU/contract 做 cache-local 分块，scratch 缓冲缩容不再清零，
field pool/merge 和 4 通道 head 使用 AVX2；d0/full 的小通道窗口块进一步在每个窗口内完成
FFN→QKV→attention→projection，避免对高分辨率 field 重复扫内存。优化快速路径只在未开启
`--dump-dir`/`NR_PROFILE` 时启用，因此调试转储仍走逐阶段实现。
MoE 路径按专家一次性重排整块输入，复用连续 scratch 完成 expand/narrow/qkv/proj；attention
直接写回专家顺序，去掉每个专家循环中的整场 gather/scatter。

## VNNI / INT8

`NR_VNNI=ON` 构建 256 位 AVX-VNNI 和 512 位 AVX-512 VNNI 微内核。
CMake 会检查编译器的 intrinsic 支持；不可编译的内核被跳过。
每种 ISA 单独编译，运行时检查 CPUID、OSXSAVE 和 XCR0；
默认优先 AVX-512 VNNI，其次 AVX-VNNI，再回退到既有 AVX2/scalar。
VNNI 加速 INT8 点积，默认 FP32 推理需显式启用 `NR_INT8=1` 才会使用它。

```powershell
$env:NR_INT8 = '1'
$env:NR_VNNI = 'auto'
engine/build/Release/cpu_engine.exe teacher/samples/lake.png --weights weights/student_mixed `
    --style 0 --tone 0.5 --structure 0.5 --skin -1 --no-automask `
    --size 512 --threads 8 --bench --warmup 10 --repeats 50 -o out/lake_vnni.png
```

日志会显示 `mixed-int8/avx-vnni`、`mixed-int8/avx512-vnni` 或实际回退后端。
`NR_VNNI=0` 关闭 VNNI；`avx` / `avx512` 可分别选择一种 VNNI 做性能对比，
指定的 ISA 不可用时仍安全回退。编译时完全去除 VNNI 可设置 `-DNR_VNNI=OFF`。
移除 `NR_INT8` 环境变量恢复 FP32。

VNNI 权重在加载期按 16 个输出、4 个 K 元素交错打包，专家层逐专家独立打包。
内核使用不饱和 `VPDPBUSD`、INT32 累加；把有符号激活映射为 `u8 = s8 + 128`，
用加载期计算的 `-128 * sum(weight)` 修正偏移，保持原 INT8 路径的量化、scale、
bias、SiLU 和残差语义。前向不重新打包权重，动态激活量化仍计入前向耗时。
当前模型沿用混合 INT8 的 `K >= 256` 策略，小 K 层继续 FP32。
VNNI 路径对低分辨率/VIT 的少行矩阵缩小行块，以保持工作线程的并行度。

`student_moe_v0` 的 C++ 路径复刻 Python 的窗口/Token top-1 路由：router 前置
LayerNorm、严格 quota 平衡、picked probability × expert gate、专家 expand/SiLU/narrow，
以及按同一专家分组的 qkv/proj。窗口路由使用未移位 8×8 窗均值，attention 仍使用块相位的
移位窗；边界窗口只统计有效像素。

本机 i9-10900K 仅支持 AVX2，无法实测 VNNI 硬件耗时。
相同 8 线程、10 次预热、50 次测量下，AVX2 混合 INT8 的 320/512 前向
median 为 30.593/77.133 ms，p90 为 34.480/83.090 ms；本机保持默认 FP32。
`vnni_check` 覆盖极值、零行、K/M/N 尾块、权重修正、bias/SiLU/残差和原地输出。
MSVC AVX2、Clang AVX2 与纯标量构建均通过 CTest；Clang 关闭隐式浮点融合，
确保 field 标量尾部与 SIMD 的乘法/加法舍入一致，显式 GEMM FMA 仍保留。
Intel SDE 的 Alder Lake、Sapphire Rapids
模拟执行验证两种 VNNI 指令路径，GEMM 输出与 AVX2 INT8 逐位一致。
真实 `student_mixed` 权重的 320×320、512×512 head 二进制和 NR PNG
在三种后端下也逐位一致。
模拟器耗时不能用作真实性能数据。

```powershell
ctest --test-dir engine/build -C Release --output-on-failure
# 在支持 VNNI 的 CPU 上要求测试必须执行 VNNI（不可用时测试失败）：
engine/build/Release/vnni_check.exe --require-vnni
```

## 加载与数值口径

发布包加载沿用 `teacher/dump_weights.py` 的张量布局和块配置：
读取 `manifest.json` 的 f32 张量、拼接 `model/stages/*.bin`，
按 `shape.json` 生成 enc/dec 数量、expert/split 参数和窗口相位，再复用原来的权重打包与前向。
加载时校验 stage 长度、tensor 范围/shape/dtype；当前 C++ 加载器不计算 SHA256。
本次验证用 `teacher/nr_package.py::read_package` 检查了发布包全部 stage 的 SHA256。

出图采用现有首帧语义：`neural = clamp(proxy + head_rgb / 4, 0, 1)`。
第四通道是历史帧混合权重，单张图片没有历史帧，因此不参与 RGB 合成。
`--blend-png` 可输出该权重图，`--proxy-png` 可输出缩放后的输入。

验证结果：

- 320²/512² 下，发布包直读与 `dump_weights.py` 导出的 idx 加载得到的 head
  和 60 份中间捕获均逐位一致。
- `teacher/check_engine.py --bisect` 对账通过，最终 head 最大误差分别为
  `8.702e-6`、`8.821e-6`（通过阈值 `2e-3`）。
- 实际图片入口在 320²、512²、320×512 下与相同输入的 PyTorch 前向一致，
  lane 对账通过，PNG 合成逐像素一致；JPEG、中文路径、原尺寸和重复运行确定性通过。
- `engine/build/Release/kernel_check.exe` 覆盖 FFN/expert 融合、原地残差、4 通道 head、
  padding/tail channel；AVX2=ON 和 AVX2=OFF 两种构建均通过 CTest。

校验环境使用 `uv venv .venv --python 3.10` 建立，Python 只用于数值校验。
例如以下命令仍可检查原 idx 路径（`checkpoint.pt` 需匹配该 shape）：

```powershell
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '8'
.venv/Scripts/python.exe teacher/check_engine.py --image teacher/samples/lake.png --size 320 `
    --shape weights/student_mixed/shape.json --checkpoint checkpoint.pt `
    --engine-build engine/build/Release --bisect
```

## 当前 MoE 与 CNN 学生

CPU 和原生 CUDA 均可从包内 `shape.json` 自动识别：

- `nr_shared_moe.py`：当前 7 专家 Top-1、无共享分支，支持 `vit.blocks=0`、
  窗口膨胀及冻结的 router bias。gate 为 `E * softmax(logits)[pick]`，子层融合系数为 0.5。
  也保留旧的 1 个共享 + 16 个路由专家 Top-1 包。
- `nr_cnn_unet.py`：完整七级 CNN U-Net，支持 3×3/1×1 卷积、8 组 GroupNorm、
  `align_corners=False` 双线性缩放、拼接跳连和像素级 8 专家 FFN。
  CNN 的 gate 为 `softmax(logits)[pick]`，保留专家的两层 bias 和独立 SiLU。

两条新路径使用 FP32（关闭 TF32），CPU 不支持 `NR_INT8=1`。
历史 Top-2 shape 不属于当前 Python 模型契约，引擎会明确拒绝。

先在有 PyTorch 的环境打包 checkpoint，再把目录交给现有图片/视频入口：

```powershell
python teacher/make_student_package.py --shape shapes/cnn_unet_moe8.json `
    --checkpoint runs/cnn/ckpt.pt -o weights/cnn
# MoE 使用 --shape shapes/student_routed7_v1.json
engine/build-cuda/gpu_engine.exe photo.jpg --weights weights/cnn -o out/cnn.png
engine/build/Release/cpu_engine.exe photo.jpg --weights weights/cnn -o out/cnn_cpu.png
```

`convert_weights.py --shape student` 和 `check_package.py` 也支持两种模型。
重新构建后，下面的检查无需训练权重，会导出完整生产 shape 并验证四种尺寸、
逐层输出、路由选择、CUDA eager/Graph/repeat 一致性：

```powershell
.venv/Scripts/python.exe teacher/check_native_students.py
.venv/Scripts/python.exe teacher/check_native_students.py --edge-cases --sizes 320x320 320x512
engine/build-cuda/shared_moe_check.exe --quick
```

可用 `--package weights/cnn` 检查自己的包，或 `--cpu-only` 只测 CPU。
完整模型算术测试会轮换 CNN 各块的优势专家以避开 argmax 临界点；
独立 CUDA fixture 覆盖混合路由、并列取最小索引、偏置选择、空专家和非整齐通道。
真实权重若出现几乎相同的 logits，FP32 运算顺序差异可能让少数像素换专家，
此时逐元素严格对账会报错，需要结合最早出现差异的路由判断。

旧共享原型的设计与历史测量见 [`teacher/SHARED_MOE.md`](../teacher/SHARED_MOE.md)
和 [`teacher/SHARED_MOE_REPORT.md`](../teacher/SHARED_MOE_REPORT.md)。
