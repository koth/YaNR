# distill-training Specification

## Purpose
TBD - created by archiving change fast-student-distillation. Update Purpose after archive.
## Requirements
### Requirement: 伪量化训练
训练 SHALL 从第一步起注入 fp8(e4m3)/f16 舍入语义(直通估计),使学生在推理数值下工作;QAT 开关 SHALL 可用于消融。

#### Scenario: QAT 开关
- **WHEN** 分别以 QAT 开/关跑同一 sanity run
- **THEN** 两次运行均可完成,量化开关带来的质量差异被记录

### Requirement: 四类损失体系
训练损失 SHALL 覆盖输出(rgb/4 残差 L1 + blend logit SmoothL1 + 合成图 L1)、细节(拉普拉斯/Sobel 高频)、特征(逐级 cosine+L2,对齐在伪量化激活上)与时序(帧间差分一致)四类,权重可配置。

#### Scenario: 损失分量可追溯
- **WHEN** 训练每 1K 步记录日志
- **THEN** 四类损失分量分别可读,任一分量可单独置零做消融

### Requirement: 训练运行与断点续训
run v1 SHALL 在单卡 3090 上 ≤72h 完成并支持断点续训(权重/EMA/optimizer/step 全恢复);过拟合测试(单 batch 2K 步)SHALL 达到学生-教师合成图 PSNR ≥ 40 dB 后才允许启动全量训练。

#### Scenario: 中断恢复
- **WHEN** 训练进程被杀后以 --resume 重启
- **THEN** 从最近 checkpoint 继续,损失曲线与不中断运行一致(同 seed)

#### Scenario: 过拟合门禁
- **WHEN** 运行单 batch 过拟合测试
- **THEN** 2K 步内合成图 PSNR(学生 vs 教师)≥ 40 dB,否则训练管线判为未就绪

### Requirement: 消融实验
训练工具 SHALL 支持 -特征蒸馏 / -时序 / -QAT / -细节损失四组消融短跑(各 20K 步)并产出对比表。

#### Scenario: 消融产出
- **WHEN** 运行四组消融并汇总
- **THEN** 产出含 PSNR、blend MAE、时序指标的对比表,指明各损失项的贡献

