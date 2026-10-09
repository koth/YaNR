# Proposal: fast-student-distillation

## Why

[teacher/](teacher) 里的 DLSS5 NR 复原网络(71 块、310.8.x 真权重)在 3090 上 512² 单帧 113 ms、1080p 约 720 ms,离产品化很远。kernel 已贴近 fp8 张量核上限(~600 G products/s),内存占比 <7%,继续融 kernel 只能再拿几个百分点;实测成本地图显示 7 个级各占 12–17%,想快 5 倍只能把 MACs 整体砍到 1/5。因此要蒸馏出一个小模型:同几何、同 16 lane 输入、同 4 通道 head 与 composite 语义,宽度减半、深度收缩、ViT 大砍,把 512² 压到 ~25 ms,同时保住教师的渲染性格(±0.03 残差、sigmoid blend)。

## What Changes

- 新增 `cost-estimator` 能力:逐块 GMAC 记账、实测校准的 ms 预估、学生形状 DSL 与批量扫描,把"5× 从哪来"变成可查账的数字。
- 新增 `student-network` 能力:学生形状配置、学生网络骨架(torch 路径先行)、与 run_image.py 兼容的推理入口、蒸馏特征捕获点。
- 新增 `distill-data` 能力:图像/序列数据、proxy 退化管线、history lane 语义 spike、教师标注缓存(逐级激活 + head)。
- 新增 `distill-training` 能力:QAT 伪量化蒸馏训练(输出/细节/特征/时序四类损失)、断点续训、消融实验。
- 新增 `student-eval` 能力:质量(对教师/对输入 PSNR、SSIM、时序稳定)与性能(≥5×)验收与报告。
- 新增 `student-packaging` 能力:学生权重打包为 manifest + stages、推理加载兼容、文档与许可声明。
- 新增 `cpu-engine` 能力:CPU 推理引擎(ONNX Runtime + int8)、16 lane 输入管线的 C++ 移植、Windows 消费机 30ms 性能验收(512² 口径)。
- 不修改 teacher 现有数值行为与校验语义;学生代码全部是新增文件,或带开关的新增路径;teacher 校验套件必须保持全绿。

## Capabilities

### New Capabilities

- `cost-estimator`: 逐块 GMAC 成本模型、实测校准、学生形状 DSL 与扫描选型。
- `student-network`: 学生网络结构、形状配置、torch 前向与推理入口。
- `distill-data`: 蒸馏数据生成、proxy 退化、history 语义确认、教师标注缓存。
- `distill-training`: QAT 蒸馏训练、四类损失、训练运行、断点续训与消融。
- `student-eval`: 质量/时序/性能验收口径与验收报告。
- `student-packaging`: 学生权重打包、加载兼容、文档与许可声明。

### Modified Capabilities

无 —— 现有能力的需求均不变化。

## Impact

- 新增文件:`teacher/cost_model.py`、`teacher/nr_student.py`、`teacher/run_student.py`、`teacher/train_distill.py`、`teacher/eval_quality.py`、`teacher/annotate.py`、`teacher/degrade.py`、`shapes/student_v0.json`,以及 [teacher/convert_weights.py](teacher/convert_weights.py) 的 `--shape student` 分支;[teacher/check_fused.py](teacher/check_fused.py) 增加学生形状用例(新增测试,不改旧断言)。
- 依赖:torch(训练)、PIL/numpy(已有);SSIM/LPIPS 优先用 torchvision,走代理装不上则自实现 GMSD 兜底。
- 训练在服务器 `koth@192.168.3.32`(单卡 3090,代理 `http://192.168.3.127:7897`),数据与训练权重不入库。
- 不触碰 `weights/nvngx_dlssnr.dll` 与提取权重的许可约束;学生权重是蒸馏产物,README 声明不含 NVIDIA 权重、不得随源分发教师权重。
