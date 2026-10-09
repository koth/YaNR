# student-network Specification

## Purpose
TBD - created by archiving change fast-student-distillation. Update Purpose after archive.
## Requirements
### Requirement: 契约与几何不变
学生网络 SHALL 复用 teacher 的几何规则(field/padding/窗口相位)、16 lane 输入语义、32→4 f16 head 与 composite 语义(clamp(proxy + rgb/4)、sigmoid(blend)),不得改变任何一项的数值定义。

#### Scenario: 推理入口兼容
- **WHEN** 用 run_student.py 与 run_image.py 对同一输入图、同 seed 推理
- **THEN** 两者 CLI 参数、输出文件名与 composite 公式一致,仅网络权重不同

### Requirement: 学生形状与前向实现
学生网络 SHALL 由 shapes/*.json 驱动(通道/hidden 为 32 倍数,heads = ch/32 ≥ 1),支持 plain FFN 块、expert FFN 块、split 块、FFN-only 全分辨率块与 ViT 块,前向保持教师的 f16 残差种子与 aux 缩放语义。

#### Scenario: 随机权重 smoke
- **WHEN** 以随机权重跑 320² 与 512² 前向
- **THEN** 输出形状正确、无 NaN,残差幅值与 blend 分布统计被打印

### Requirement: 蒸馏捕获点
学生网络 SHALL 提供 capture 钩子与 teacher↔student 捕获点对齐表(含 full/d0 合并时的映射规则),供训练侧做逐级特征对齐。

#### Scenario: 对齐表可用
- **WHEN** 训练代码按对齐表请求一对师生特征
- **THEN** 返回同语义层级的两个激活张量,形状由各自通道数决定且行数一致

### Requirement: kernel 兼容与数值一致
学生形状 SHALL 全部走 nr_kernels.cu 融合路径(FFN-only、expert E∈{1,2,4,8}、split E4),且 check_fused 的学生形状矩阵与 torch 参考一致;融合路径占推理时间 SHALL >90%。

#### Scenario: 学生形状融合校验
- **WHEN** 运行 check_fused.py 的学生形状矩阵
- **THEN** 全部用例 PASS,3090 profile 显示 >90% 时间在融合 kernel

