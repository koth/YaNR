# 独立单帧 CNN：56M 总参数 / 7M 激活参数

本实验先学习首帧的教师 RGB 输出，不学习时序融合。模型为新增的
`nr_cnn_single.py`，入口为 `train_distill_single.py`；不修改既有 CNN/MoE
模型实现。结构配置在 `shapes/cnn_single_56m.json`，训练配置在
`remote/run_cnn_single_56m.json`。

## 结构与参数口径

| 项目 | 参数数 |
| --- | ---: |
| 全模型 | 55,999,989 |
| 公共部分，包含卷积、归一化、router、head 等 | 3,699,189 |
| 16 份专家的总和 | 52,300,800 |
| 每块选一份专家的总和 | 3,268,800 |
| 单位置逐层激活等效参数 | **6,967,989** |

因此是 `3.699M + 52.301M / 16 = 6.968M`，约为总参数的 1/8。
16 选 1 只作用于专家 FFN，公共部分始终执行。不同像素可以选择不同专家，
整图可能访问全部专家；这个数字不等于整图参数访问量、FLOPs、速度或显存比例。
训练用特征投影器另有 797,696 个参数，不随推理模型导出。

本机实测见 [速度比较](SINGLE_FRAME_SPEED.md)：当前 PyTorch 前向与旧 CNN
大致同档，尚未取得明显提速；参数预算不能代替性能和画质验证。

七级 U-Net 通道为 32/32/64/128/192/256/384。空间分支使用深度卷积与
1×1 通道混合；中低分辨率级交替使用 1/2/3 膨胀率。每像素经过通道
LayerNorm，不跨空间或 batch 统计，避免 GroupNorm 统计量随 crop 内容变化。
这不保证裁剪推理与整图推理完全相同：边界、下采样对齐和上下文仍会影响输出。

每块的 FFN 为 16 专家 top-1，按实际选中行执行，没有执行 16 个专家再相加。
gate 保留选中专家的 softmax 概率，未重归一成常数 1，router 能从主任务取得梯度。
没有共享 FFN 专家，公共空间分支始终执行。使用残差缩放和小幅 head 初始化，
让初始输出接近输入。保留 16 lane 和四通道 head 契约；本实验第四通道与
`blend_scale` 不参与时序监督，不能据此认为模型已有时序能力。

训练启用确定性算法。双线性缩放采用等价的半像素坐标、可分离索引与插值，
避免原生 CUDA bilinear backward 在严格确定性模式下报错；前向与梯度已与
`F.interpolate(..., align_corners=False)` 做容差对账。该实现增加的临时张量
和索引调度仍需在正式服务器测量，不把稀疏参数数目当成速度结果。

## 近期 CNN 研究与本版取舍

查阅日期为 2026-10-19。论文任务、数据和评价与本项目不同，论文提升不能
直接作为本项目的质量承诺。

| 来源 | 有用的方向 | 本版处理 |
| --- | --- | --- |
| [ArtIR / TIP，2026 年 10 月](https://signalprocessingsociety.org/index.php/publications-resources/ieee-transactions-image-processing/2026/10/large-kernel-network-efficient-image-restoration) | 通道压缩后的自适应大核及多尺度融合，降低扩大感受野的成本 | 作为后续空间分支消融候选；未声称复现 ArtIR |
| [DarkIR / CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Feijoo_DarkIR_Robust_Low-Light_Image_Restoration_CVPR_2025_paper.html) | 恢复任务中的高效 CNN、多尺度膨胀深度卷积 | 采用交替膨胀率扩大中低分辨率感受野；并非其并行分支或频域结构的复现 |
| [NAFNet 官方实现](https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py) | 通道归一化、深度与点卷积、残差缩放 | 借鉴稳定训练原则；本版仍使用 SiLU 和稀疏 FFN，未称为 NAFNet |

先建立能复现的数据、验证与路由基线，再单独替换空间块比较大核/门控。
同时改变数据、loss 和结构只能评价整套新方案，不能证明是哪项带来收益。

## 数据准备：不生成新退化数据

原始图片保持原样。新清单排除路径组件 `proxy_proc`，不删除历史数据。
不运行 `degrade.py` 或 `gen_sequences.py`。教师与学生共享**同一份变换后输入、
同一噪声 seed 和同一组 controls**；history=None，lane 7–9 复制当前图 lane 4–6。
这是“在同一输入上模仿教师”，不是带退化输入与无退化标签的配对恢复。

采样首先选模式，然后在符合条件的来源中按权重选来源、均匀选 group、均匀选帧。
来源权重为 DIV2K/Flickr2K/Full-Resolution 各 15%，rgb 10%，gameplay 15%，
portraits 25%，frames_2k 5%。ROI 模式仅从有 ROI 且尺寸足够的来源中选取，
会重新归一权重，因此这些值不是整个训练流的最终来源百分比。实际分布见日志。

| 模式 | 概率 | 变换 |
| --- | ---: | --- |
| ROI 原尺度 crop | 40% | 围绕已索引框中心抖动裁剪，不 resize |
| 随机原尺度 crop | 20% | 在原图直接裁剪，不 resize |
| 缩放 crop | 20% | 等比尺度变换后裁剪，包含最高 1.5 倍视野 |
| 全图上下文 | 20% | 等比缩进长宽比桶，保留真实有效矩形 |

crop 为 320/512，概率 7:3。原尺度 crop 不从尺寸不足的图中放大取样。
ROI 在 512 无可用图时会使用较小的可用 crop，并如实记录尺寸；完全没有合格
ROI 时明确失败。上下文桶为 320²、512×320、320×512、640×384、384×640、512²，
不把横竖图拉成正方形、不添加黑边作为监督内容，剩余 padding 沿用教师镜像语义。
缩放 crop 为保证较小图片可用，可能进行等比放大；该行为有 scale 元数据。

准备时解码检查、检查最小边 33、过滤极端长宽比、逐文件 SHA256 去重，按 group
划分 5% holdout。重复图片连接的不同 group 在去重前合并，避免剩余视频帧泄漏。
这是精确文件去重，不是近重复图像检测。

**目录分组是需确认的数据约定。** 静态图片默认按文件分组；视频类来源按父目录
分组并每 8/16 帧保留一帧。必须保证一段视频/场景不会散落到多个目录。
同一人物或视频跨目录、跨来源时，提供显式 group 文件，例如：

```jsonl
{"path":"portraits/person_a/01.jpg","group":"identity/person_a"}
{"path":"rgb/clip_01/001.png","group":"identity/person_a"}
```

使用 `--groups-jsonl` 传入，group 为跨来源共享名字，path 相对 data_root 或绝对路径。
默认清单不能自动发现同一人物或不同编码的视频近重复。合成序列对应同一静态
原图时也应使用相同 group。

ROI 可以使用已有的人脸/皮肤框 JSONL：

```jsonl
{"path":"portraits/a.jpg","rois":[[100,120,500,600]]}
```

坐标是 EXIF 旋转后的原像素 x0/y0/x1/y1。或者使用 `--detect-faces`，需要环境已
安装 `opencv-python-headless`，离线运行 OpenCV 自带 Haar 人脸检测，不下载模型。
它覆盖不了侧脸、皮肤或所有游戏人物，清单有 ROI 数量与拒绝原因，应该先审查
覆盖率。索引脚本不会把整张人像图伪装成人脸标注。

```bash
cd /data2/hezm/klss/code
source /data2/hezm/klss/env/bin/activate
python3 teacher/prepare_single_data.py \
  --config remote/run_cnn_single_56m.json --detect-faces
# 有更可靠的标注时改用 --roi-jsonl /path/rois.jsonl，可附加 --groups-jsonl。
```

所有配置中的根目录是远端布局。准备脚本支持 --data-root 和 --out 覆盖；训练
支持 --manifest 覆盖。本次未能重新读取远端全量原始数据，七个来源的存在性、
实际重复率、ROI 覆盖率仍应以准备脚本的扫描结果为准。

## Loss 与固定验证

首轮配置偏向像素位置准确性和 PSNR，而非只匹配纹理幅度：

| 项 | 配置 |
| --- | ---: |
| 残差/合成图 L1 等 output 项 | 1.0 |
| 合成图 MSE（0–1 RGB） | 100.0 |
| Laplacian + Sobel | 0.25 |
| 高频位置 L1（位于 output 项内部） | 1.0 |
| 逐级特征 | 0.05 |
| 纹理能量统计 | 0.0 |
| 时序、blend logit、pair、在线退化 | 全部 0 |
| MoE balance / router z-loss | 0.01 / 0.001 |

沿用旧 loss 函数的动作/高频加权语义。MSE 是新增的全有效区域无权重项，
弱改动区域也有直接监督；100 是补偿 MSE 与 L1 的量纲差，不代表其实际梯度
必定占主导。特征项从旧 CNN 常用 0.5 降到 0.05，减少异构网络逐层强对齐的约束。
这些系数是首轮假设，尚无训练结果证明最优。若 PSNR 与观感冲突，保留固定
清单、同输入噪声，对 MSE/高频/特征做单项消融，避免仅凭总 loss 调权。

每 2,000 步在固定 holdout 中的同 16 张图上验证，EMA 权重，分别报告固定中心
crop 与原尺寸整图。每张图记录 noise seed、变换、有效区域、学生对教师 PSNR、
输入对教师 PSNR，以及已有 ROI 内两项 PSNR。均值是逐图 dB 的算术平均；ROI
均值仅覆盖有可见框的图，框是矩形区域，并非精确皮肤分割。原尺寸 field 超过
3,145,728 行时显式标记跳过，不缩小后冒称原尺寸。16 张固定清单是初步验证，
后续应扩充到按来源、场景与人脸大小分层的独立评测。

`log.csv` 的 loss 为实际加权训练目标；MSE、out、detail、feature 等列为各分项
未乘外层系数的数值。Comet 保留 psnr/teacher 训练曲线，并另报 val/crop 与
val/native。`sampling.jsonl` 记录每个日志窗口全部模式/来源计数及当步样本；
`routing.jsonl` 记录最后训练 microbatch 的逐层专家计数、最大占比和概率均值，
不能把小尺寸深层偶尔未用全专家直接判为训练塌缩。

## 启动、显存与恢复

仅占用远端物理 GPU 6；默认关闭教师/学生重叠以降低峰值。batch 10 通过按
有效矩形分组、按 field 行预算分块累积。学生与教师行预算各 589,824；512²
实际 field 为 576×512，一次最多两张。单张超过预算会报错，不暗中缩放。
CPU 预取有界，教师只保留一份权重，在串行锁内切换几何。

先运行 GPU 模型冒烟，再用独立输出目录跑真实数据 2 步：

```bash
export CUDA_VISIBLE_DEVICES=6 NR_CUDA_ARCH=sm_80
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVCC=/usr/local/cuda/bin/nvcc
python3 teacher/check_single_model.py --device cuda --backward
python3 teacher/train_distill_single.py --config remote/run_cnn_single_56m.json \
  --steps 2 --batch 2 --val-count 2 --comet-disabled \
  -o /data2/hezm/klss/runs/cnn_single_56m_smoke
```

正式启动可直接从 code 目录执行：

```bash
CONFIG=/data2/hezm/klss/code/remote/run_cnn_single_56m.json \
RUN=/data2/hezm/klss/runs/cnn_single_56m_run1 \
bash remote/start_train_single.sh
```

脚本会检查 GPU 6 无计算进程，不停止既有训练；已有 run 目录不自动覆盖。
断点恢复直接调用 trainer，使用原正式配置与目录：

```bash
python3 teacher/train_distill_single.py --config remote/run_cnn_single_56m.json \
  --resume /data2/hezm/klss/runs/cnn_single_56m_run1/ckpt.pt \
  -o /data2/hezm/klss/runs/cnn_single_56m_run1
```

保存模型、特征投影、优化器、EMA、采样索引、Torch RNG，以及教师 manifest、
数据清单、结构和关键源码 SHA256。恢复时检查来源哈希与采样配置，防止改代码
或换标签后悄悄续训。精确数值复现还依赖相同软件、硬件与原始图片未被改写。

Windows 已有 GPU 环境：`tmp/teacher-cuda-venv/Scripts/python.exe`，
Torch 2.6.0+cu124 / RTX 3080。仓库 `.venv` 是另一套 CPU Torch 环境。
已有教师 Windows 库可以通过 `NR_CUDA_LIBRARY` 指向
`tmp/shared_moe/nr_kernels.dll`；新 trainer 会记录该库的哈希。Windows 本地
模型检查使用物理 GPU 0，远端 GPU 6 限制不适用于本机 RTX 3080。

新入口在 Linux 按 `NR_CUDA_ARCH` 编译独立的教师库缓存，文件名包含 CUDA
架构与 kernel 源码哈希，A100 使用 sm_80；不依赖旧 loader 的 sm_86 编译默认值。
显式提供 `NR_CUDA_LIBRARY` 时使用给定文件并记录其哈希；Windows 未指定时
自动使用上述已存在的 DLL。

## 测试与导出边界

```bash
python -m unittest discover -s teacher -p test_single_frame.py -v
python teacher/check_single_model.py --backward --out tmp/single_frame_model_check.json
python teacher/convert_weights.py --shape student RUN/ckpt.pt \
  --shape-json shapes/cnn_single_56m.json -o weights/cnn_single_56m
python teacher/check_package.py --package weights/cnn_single_56m \
  --checkpoint RUN/ckpt.pt --shape shapes/cnn_single_56m.json --size 320
python teacher/run_image.py IMAGE --weights weights/cnn_single_56m \
  --width WIDTH --height HEIGHT --style 2 --tone 1 --structure 1 --skin -1 --automask -o out_single
```

Python 包加载已支持新 architecture；保持首帧推理，**不传 --prev**。
现有 C++ 引擎尚未实现这个新增 architecture，不能直接拿新包走 C++ 部署。
这里没有改动旧 CNN/MoE 的参数命名、数值或推理代码。

CPU 测试覆盖参数实数、路由执行/梯度、归一化、插值与反向、矩形 batch/loss、
打包往返、去重分组、显式身份分组、原尺度像素、上下文长宽比、线程顺序确定性、
ROI 失败条件、EMA 验证恢复与超预算报告。真实教师短步冒烟只能证明链路能运行，
不能证明正式收敛、整图泛化或相对 run6 的质量提升。

### 2026-10-19 本地验收

- 14 项新 CPU 测试通过；Python 3.10 / Torch 2.14.1+cpu。
- 切换已有 CUDA 环境后，完整 56M 模型通过 320²、512²、640×384、384×640、
  512×288 前向与 320² 反向检查，输出和梯度有限；Torch 2.6.0+cu124 / RTX 3080。
- 使用本地真实教师 `weights/teacher` 跑完 2 步（batch 2、320²、microbatch 1），
  每步验证并保存；随后恢复 checkpoint 再跑 1 步成功。样例仅作链路夹具，
  训练/验证样例取自 t1–t7，不是正式 holdout，也没有质量提升结论。
- 恢复后的 EMA checkpoint 导出 224.00 MB f32 包，2,682 个 tensor 权重往返
  逐位相等，320² 前向输出逐位相等。产物位于 `tmp/single_frame_pipeline_final/`。
- 教师 `check_numerics.py --torch` 与 `check_fused.py` 在现有 CUDA DLL 下通过；
  `check_model.py weights/teacher`、12 项旧 cost-model 测试、320² lane/history
  对账和 `check_engine.py --bisect` 通过。Windows 旧 shape 文本测试使用
  `python -X utf8`；新文件显式按 UTF-8 读取。
- 新启动脚本通过 `bash -n`。没有在服务器启动正式训练，也未验证 Linux/A100
  上的编译、峰值显存与吞吐；下一步需先准备实际数据清单并跑服务器短步冒烟。
