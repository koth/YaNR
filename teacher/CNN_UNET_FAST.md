# CNN U-Net 优化版

以既有 `nr_cnn_unet.py` 为基线，新增 `nr_cnn_unet_fast.py`，不改旧模型实现。
保留 16 lane 输入、四通道 head、七级 U-Net、GroupNorm、3×3 空间卷积、
8 专家像素 top-1 和原 softmax gate。它不是上一版 16 专家 / 7M 激活模型。

| 项目 | 旧 CNN | 优化版 |
| --- | ---: | ---: |
| 训练参数 | 62,116,253 | 56,287,389 |
| 融合后推理参数 | 62,116,253 | 55,358,397 |
| 单位置激活等效参数，推理 | 22,519,549 | 21,102,301 |

推理参数减少约 10.9%，训练参数减少约 9.4%。本版优先保留旧 CNN 的容量与
空间建模方式，不再把 7M 激活当成目标。参数统计包括 bias、router、norm、
head；不包括训练时另加的教师特征投影器。激活等效参数不是显存或 FLOP 比例。

## 改动

### 专家执行

旧版每块调用 8 次 CUDA nonzero，再分别 gather、Linear、scatter。
默认新执行器 `expert_execution=split` 对所有像素稳定排序一次，一次性读取
8 个组长度，然后用连续视图执行各组的 Linear，拼接后用逆排列恢复位置。
35 个块的动态分组长度同步从 280 次减少到 35 次；不改变路由粒度、gate、
专家数量，不丢 token，不设置容量上限，也不计算全部专家后再掩码。

组长通过排序后的边界搜索获得，避免大量像素对少数计数器进行原子写入。
专家矩阵保留旧 Linear 的连续切片布局；反向用一次 unbind 汇集整份 bank
的梯度，避免每个专家都创建整块梯度缓冲。另有 `auto` 模式，可在超过
`split_max_rows` 时回退逐专家路径；正式默认仍使用实测较快的 split。

另保留 `expert_execution=tiled`：紧凑瓦片分组 baddbmm，适合做速度/显存消融。
它会产生补齐与逐瓦片权重临时张量；默认 split 避免这些开销。两种执行器
均与逐专家参考做输出及梯度对账，不能把某一模式的数字当作另一模式。

Packed bank 的未选专家切片梯度为零，旧 ModuleList 未选专家可能为 None。
因此 AdamW 对空专家的衰减/动量更新不与旧 optimizer 逐位等价；旧优化器
状态不能直接迁移后声称无缝续训。同权重前向对照只用于验证执行语义。

### 结构和初始化

- full / d0 / d1 / d2 的空间卷积训练时有 3×3、1×3、3×1、1×1 四个线性分支，
  推理前将核按中心补齐相加、bias 相加，融合为一个普通 3×3。额外分支从零
  开始，保留主分支的初始行为，同时有非零分支权重梯度。
- 深层 d3/d4 的 expert hidden 从 512 降到 448，d5 从 960 降到 832，d5 通道
  从 512 降到 480。其余通道、块数和 decoder 拼接方式保持基线设计。
- 新结构的 head 使用标准差 0.001 的小权重和零 bias，让初始 RGB 输出接近
  输入。输入对教师的 PSNR 同时作为验证基线，避免把接近输入误认成蒸馏收益。
- 上采样前向直接调用原生半像素 bilinear。自定义反向按两个轴做有序分段
  求和，避免严格确定性 CUDA 下 bilinear backward 报错。没有改变有效区域
  或教师几何；CPU 的输出逐位对账、梯度容差对账，GPU 另做冒烟检查。

### 查阅的研究与取舍

查阅日期：2026-10-19。下面是设计参考，不是本项目已经达到 SOTA 的证据。

| 一手来源 | 本版采用/取舍 |
| --- | --- |
| [AIM 2025 高效真实去模糊报告，Fig.3 ConvRep3](https://openaccess.thecvf.com/content/ICCV2025W/AIM/papers/Feijoo_Efficient_Real-World_Deblurring_using_Single_Images_AIM_2025_Challenge_Report_ICCVW_2025_paper.pdf) | 采用训练多分支、推理单卷积的重参数化思想；不是 NAFRepLocal 的复现，也不引用其成绩作为本项目收益 |
| [NAFNet 官方实现](https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py) | 检查门控、归一化、残差设计；此次保留既有 GroupNorm 与 SiLU，避免同时替换整个恢复块 |
| [DarkIR，CVPR 2025 官方实现](https://github.com/cidautai/DarkIR) | 查看高效 CNN 恢复方向；本版未增加频域分支或并行膨胀卷积，以免增加部署算子与消融变量 |
| [ArtIR，TIP 2026 年 10 月](https://signalprocessingsociety.org/index.php/publications-resources/ieee-transactions-image-processing/2026/10/large-kernel-network-efficient-image-restoration) | 最新大核/通道压缩方向的参考；当前未复现其自适应单通道大核模块，不将普通卷积标成 ArtIR |

## 训练、验证与启动

准备正式开训时，先按 [开训交接说明](../remote/TRAIN_CNN_UNET_FAST.md) 核实路径、
生成独立配置、准备离线人脸框与数据清单，再完成服务器短步验收。下面的远端路径
仅为示例；启动脚本的 ROOT 不会自动改写 JSON 内部路径。

形状在 `shapes/cnn_unet_fast.json`，配置在 `remote/run_cnn_unet_fast.json`。
复用新的单帧数据清单、原尺度/ROI/上下文裁剪、去重与分组 holdout，以及
PSNR 导向的单帧 loss；不启用时序、pair、blend logit 或退化生成。
数据规则与各 loss 系数详见 [单帧流水线](SINGLE_FRAME_56M.md)。

```bash
# 已准备 manifest 后，先独立输出目录冒烟。
export CUDA_VISIBLE_DEVICES=6 CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 teacher/check_single_model.py --shape shapes/cnn_unet_fast.json --device cuda --backward
python3 teacher/train_distill_single.py --config remote/run_cnn_unet_fast.json \
  --steps 2 --batch 1 --val-count 1 --comet-disabled -o /data2/hezm/klss/runs/cnn_fast_smoke
# 正式运行；脚本检查 GPU 6 是否空闲，不覆盖既有 run。
bash remote/start_train_cnn_fast.sh
```

50k 步配置、batch=10、320/512=7:3、按 field 行预算分块累积均是首轮设置。
Windows 本机使用 `tmp/teacher-cuda-venv/Scripts/python.exe` 和 GPU 0。
10GB 卡应根据实测降低 `batch_rows`（例如 294912）以限制单 microbatch，
完整训练峰值还包含教师、AdamW、EMA、特征投影和 loss，不由模型单独反向峰值保证。

## 导出

支持两条出口。`native` 将分支融合，并把 packed bank 展开回旧 CNN 的张量
命名与 DSL，可交给现有 C++ CPU/CUDA；`pytorch` 保留优化执行器并融合分支。
原始训练 checkpoint 保持未融合状态，供原配置恢复。

```bash
python3 teacher/export_cnn_fast.py RUN/ckpt.pt --shape shapes/cnn_unet_fast.json \
  --backend native -o weights/cnn_unet_fast_native
python3 teacher/export_cnn_fast.py RUN/ckpt.pt --shape shapes/cnn_unet_fast.json \
  --backend pytorch -o weights/cnn_unet_fast_torch
engine/build-cuda/gpu_engine.exe photo.jpg --weights weights/cnn_unet_fast_native -o out.png
```

默认导出 EMA，缺少 EMA 会明确报错，不静默改用 raw。用 `--state model` 可显式
导出原始训练权重。输出包已存在时拒绝覆盖。

`CNNUNetFast.from_legacy(old_model)` 可将旧 CNN 权重转为同容量的分组执行模型，
不做剪枝，因此仍是 62.12M；不能把它与需要重新训练的 55.36M 新结构混同。

```bash
python3 teacher/export_cnn_fast.py --legacy-package weights/cnn_unet_moe8 \
  --backend pytorch -o weights/cnn_unet_moe8_grouped
python3 teacher/run_image.py teacher/samples/t7/t7.jpg \
  --weights weights/cnn_unet_moe8_grouped --width 512 --height 512 \
  --style 2 --tone 1 --structure 1 --skin -1 --automask -o tmp/cnn_grouped_preview
```

## 检查与速度

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s teacher -p test_cnn_unet_fast.py -v
.\tmp\teacher-cuda-venv\Scripts\python.exe teacher/bench_cnn_fast.py --output tmp/cnn_unet_fast/speed_final.json
```

速度测量在同一张 RTX 3080、FP32、TF32 off、batch=1、相同输入与几何下进行，
三轮轮换顺序。前向每轮预热 5 次、测 15 次，前向加反向预热 2 次、测 5 次，
报告各轮中位数再取中位数。新模型推理融合分支、训练保留分支。
包含旧 CNN、同旧权重的分组执行对照、随机初始化的缩小版三个模型。
正常计时不含 teacher、IO、传输、optimizer、EMA 与蒸馏 loss。
桌面负载未完全隔离，小差异须结合各轮波动判断。

2026-10-19 本机结果（毫秒，PyTorch eager，不是 C++ 引擎耗时）：

| 有效区域 / 模式 | 旧 CNN | 同旧权重，仅优化执行 | 缩小版 | 缩小版延迟下降 |
| --- | ---: | ---: | ---: | ---: |
| 320×320 前向 | 271.01 | 138.08 | 135.02 | 50.2% |
| 512×512 前向 | 280.66 | 156.33 | 154.42 | 45.0% |
| 1280×720 前向 | 383.80 | 263.70 | 268.39 | 30.1% |
| 320×320 前向+反向 | 673.91 | 334.91 | 346.89 | 48.5% |
| 512×512 前向+反向 | 761.66 | 382.91 | 439.11 | 42.3% |

512 的 field 是 576×512，720p 的 field 是 1344×768，三个模型完全相同。
主要加速来自执行方式；减少深层参数的额外速度收益很小。缩小版训练额外
计算 Rep 分支，因此反向慢于仅优化执行的同容量对照。720p 两个优化版本
的数毫秒差异落在轮次波动范围，不能据此认为缩小参数会加速所有尺寸。

512 前向+反向峰值 allocated 显存由 4.04 降至 3.79 GiB；720p 推理则由
1.82 增至 1.89 GiB。分组排序有额外临时空间，不承诺每个尺寸都省显存。
320、512 实际旧权重 head 与保存后重载的打包版 GPU 对照最大误差均为 0；这不是与教师的
PSNR，也不是缩小版的画质评估。现有 C++ 已有专用路由算子，不能把这里的
PyTorch 执行器加速比例直接套到 C++。

另测 C++ CUDA graph（同卡、FP32、resident input、3 轮、每轮预热 5/测 15 次，
同一张 t7 输入）：

| 有效区域 | 旧 CNN / ms | 缩小版 / ms |
| --- | ---: | ---: |
| 320×320 | 46.82 | 46.70 |
| 512×512 | 135.65 | 135.82 |
| 1280×720 | 482.72 | 479.22 |

C++ 基本持平，当前不能声称部署速度显著提升。主要算力仍在保留的高分辨率
空间卷积中，少量深层收缩主要减少权重存储。旧包是 step17999，新包是三步
流水线冒烟的 EMA；路由与权重不同，此表不构成质量对照。CUDA event 口径
与前面的 PyTorch 同步 wall time 分别记录：[native_speed/report.json](../tmp/cnn_unet_fast/native_speed/report.json)。

### 验证范围

- 新模型 8 项数值单测、既有单帧 14 项测试、成本模型 12 项测试通过。
  lane（含历史/自动 mask）、旧引擎逐层 bisect、教师权重寻址、CUDA numerics
  与 fused kernel 回归通过。
- 严格确定性 CUDA 下，320、512、640×384、384×640、512×288 前向有限，
  512 反向梯度有限：[model_check_final.json](../tmp/cnn_unet_fast/model_check_final.json)。
- 使用真实教师和独立样本分组做 320、batch=1 的两步训练，保存并恢复至第三步，
  训练/EMA/optimizer 状态检查通过；完成固定 crop 验证以及 EMA 两种格式导出。
  这是流水线检查，不是收敛实验：[smoke run](../tmp/cnn_unet_fast/pipeline/run)。
- 保存后重载的同旧权重分组包、同一 EMA 的 native/PyTorch 包，在 t7 输入的
  320、512 GPU head 对账均为逐位一致：[package_parity.json](../tmp/cnn_unet_fast/package_parity.json)。
- 新通道结构的 C++ CPU/CUDA 算术夹具在 320、512 的所有边界、每块 router
  pick/gate、head 和 CUDA graph 重放通过（固定主专家以隔离算术误差，head
  最大误差 <1.6×10⁻⁵）：[native_arithmetic](../tmp/cnn_unet_fast/native_arithmetic/results.json)。

原始三步冒烟 EMA 的随机输入跨后端严格中间量测试没有全部通过，不能称为
逐位兼容：首个路由分歧的两个 logit 间隔仅 0～3.8×10⁻⁶，1～12×10⁻⁶
级别的前序算术差异改变了 top-1，随后在局部激活中放大。320 共 6 处、512
共 21～22 处 pick 不同；head 最大误差约 0.00091～0.00105、平均 <1.1×10⁻⁷。
没有修改 tie 规则或放宽断言掩盖这个失败。它说明 hard top-1 的跨后端边界
敏感性，正式训练权重还需重新对账；上面的受控算术夹具不替代这项检查。
定位记录：[native_diagnosis](../tmp/cnn_unet_fast/native_diagnosis/results.json)。

原始逐次耗时、源码/形状哈希、参数、显存与同权重误差记录于
[speed_final.json](../tmp/cnn_unet_fast/speed_final.json)。质量提升仍需正式训练和固定验证集
确认，流水线冒烟和随机权重速度不证明画质更好。
