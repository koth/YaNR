# student-packaging Specification

## Purpose
TBD - created by archiving change fast-student-distillation. Update Purpose after archive.
## Requirements
### Requirement: 学生权重打包
学生权重 SHALL 转换为与 teacher 相同的模型目录格式(manifest.json + model/stages/s*.bin,含逐 tensor 偏移与 sha256),转换后加载输出 SHALL 与训练导出的 torch 输出逐位一致。

#### Scenario: 打包往返一致
- **WHEN** 将训练导出的学生权重打包并在推理入口加载
- **THEN** 同输入下与 torch 导出参考逐位一致,manifest 校验和通过

### Requirement: 推理入口加载学生模型
run_image.py / run_nr.py SHALL 直接接受学生模型目录(形状自动探测或 --shape student),输出图 SHALL 注明所用模型版本。

#### Scenario: 端到端出图
- **WHEN** 用 run_image.py 加载学生模型目录对输入图推理
- **THEN** 产出与教师流程同名的 PNG 组(proxy/neural/blend/side_by_side),图像注明学生模型版本

### Requirement: 文档与许可声明
teacher/README.md SHALL 记录学生结构表、成本模型与扫描用法、训练/评估/打包复现命令与硬件要求;并 SHALL 声明学生权重为蒸馏产物、不含 NVIDIA 权重,教师权重与 DLL 不随源分发。

#### Scenario: 复现指引完整
- **WHEN** 新读者按 README 操作
- **THEN** 能依次跑通成本模型、训练、评估与打包命令,且许可边界清晰可见

