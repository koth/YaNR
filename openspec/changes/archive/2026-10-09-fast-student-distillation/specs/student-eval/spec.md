## ADDED Requirements

### Requirement: 质量验收口径
评估工具 SHALL 报告学生 vs 教师(合成图 PSNR/SSIM、残差相关、blend logit MAE)与学生 vs 输入(对教师 512² 基线 38.53 dB 的退化 ≤ 1.0 dB);LPIPS 依赖不可用时 SHALL 有自实现指标兜底。

#### Scenario: 数据集级质量报告
- **WHEN** 对测试集运行 eval_quality.py
- **THEN** 输出上述全部指标的数据集聚合值与单图明细,退化超限时明确告警

### Requirement: 时序稳定验收
时序稳定性 SHALL 以 warp error(块匹配光流近似)与帧间闪烁指标度量,学生指标 SHALL ≤ 1.5× 教师。

#### Scenario: 序列稳定性对账
- **WHEN** 对同一序列分别评估学生与教师
- **THEN** 输出两者的 warp error 与闪烁指标,以及学生/教师比值

### Requirement: 性能验收
3090 上 graph capture 计时 SHALL 覆盖 320²/512²/768²/1080p(30 次取中位);512² 实测加速比 SHALL ≥ 4.5×(目标 ≥ 5.0×),并报告逐 kernel 占比与成本模型预测偏差。

#### Scenario: 加速比达标判定
- **WHEN** 训练完成后运行性能验收
- **THEN** 512² 实测加速比 ≥ 4.5× 判定达标,≥ 5.0× 记为达成目标;预测偏差 >15% 时回写校准系数

### Requirement: 验收报告与回归门禁
验收 SHALL 产出 report_final.md(形状表、预算 vs 实测、质量表、时序表、消融表);teacher 校验套件(check_numerics/check_fused/check_model/check_block0)SHALL 全绿,任何学生改动不得破坏教师断言。

#### Scenario: 回归门禁
- **WHEN** 在最终验收前运行 teacher 全套校验
- **THEN** 全部 PASS(check_block0 0/32 mismatch),否则学生改动判为不合格
