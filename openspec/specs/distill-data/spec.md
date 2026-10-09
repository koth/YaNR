# distill-data Specification

## Purpose
TBD - created by archiving change fast-student-distillation. Update Purpose after archive.
## Requirements
### Requirement: 数据来源与 proxy 退化
数据管线 SHALL 提供真实高清图/视频帧序列与程序化序列两类来源,并经 proxy 退化(模糊+降采样+重建+轻噪声)生成训练输入;退化参数 SHALL 与 teacher/samples 流程的 proxy 统计(FFT/边缘能量)对齐。

#### Scenario: 训练对生成
- **WHEN** 对一批高清图或序列运行数据管线
- **THEN** 产出 proxy 帧 + 教师标注的成对样本,清单含 id/seed/sha256

### Requirement: history lane 语义确认
history lane 的来源(上帧 proxy 或上帧输出)SHALL 经专项 spike 实验确认并写入 design.md;数据与训练代码 SHALL 通过 history_source 开关支持两种语义。

#### Scenario: 序列样本的 history 填充
- **WHEN** 生成序列训练样本且 history_source 已指定
- **THEN** lanes 7-9 按指定来源填上一帧的对应内容,首帧回退为当前 proxy

### Requirement: 教师标注缓存
教师标注工具 SHALL 批量产出 head 输出(f32)与蒸馏捕获点激活(fp8/f16 存储),以分片 + 清单(含 sha256)方式缓存并支持增量生成;总存储 SHALL 控制在 100 GB 内,超预算时降级为训练时在线计算教师特征。

#### Scenario: 增量标注
- **WHEN** 数据集新增样本后重新运行标注
- **THEN** 已有分片不重复计算,新增样本并入清单且 sha256 校验通过

### Requirement: 噪声一致性
噪声 lane 的 seed SHALL 在师生之间一致(同输入同目标),并支持按 epoch 轮换;轮换策略的成本评估结论 SHALL 记录在案。

#### Scenario: 同噪声配对
- **WHEN** 以某 seed 生成样本并分别推理师生
- **THEN** 两者输入张量逐位相同,标注缓存记录同一 seed

