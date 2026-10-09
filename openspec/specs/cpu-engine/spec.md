# cpu-engine Specification

## Purpose
TBD - created by archiving change fast-student-distillation. Update Purpose after archive.
## Requirements
### Requirement: ONNX 导出与数值一致
学生模型 SHALL 可导出为 ONNX(opset ≥17),且 ONNX Runtime 前向输出与 torch 参考在同输入下最大偏差 ≤1e-3(合成图口径 PSNR 退化 ≤0.1dB);导出支持固定边长(性能最优)与动态边长两种模式。

#### Scenario: 导出对账
- **WHEN** 用 export_onnx.py 导出并在 ORT 跑同一输入
- **THEN** head 输出与 torch 参考逐元素比对,最大偏差 ≤1e-3,合成图 PSNR 差 ≤0.1dB

### Requirement: int8 量化与质量守门
引擎 SHALL 提供 int8 量化路径(先动态 PTQ,质量不足时静态 PTQ 校准或 QAT 微调);int8 引擎输出相对 f32 参考的合成图 PSNR 退化 SHALL ≤0.3dB。

#### Scenario: 量化质量验收
- **WHEN** 对测试集分别跑 f32 与 int8 引擎
- **THEN** 合成图 PSNR(int8 vs f32)≥ 30dB 且逐图退化 ≤0.3dB,否则触发 QAT 微调流程

### Requirement: 16 lane 输入管线移植
引擎 SHALL 自包含 16 lane 特征构造的 C++ 实现(Box-Muller 哈希噪声、center_proxy 三半舍入、镜像 padding、条件 lane),与 run_image.build_features 在同 seed 下逐元素一致(f16 语义口径 ≤1 LSB)。

#### Scenario: lane 管线一致性
- **WHEN** 同一输入图 + 同 seed 分别经 python build_features 与 C++ 版构造特征
- **THEN** 16 lane 张量逐元素比对通过(f16 口径 ≤1 LSB),且喂给同一模型后 head 输出一致

### Requirement: Windows 性能验收
Windows 消费机(x86-64,单机多线程)上单帧端到端(特征构造 + 模型前向)耗时 SHALL ≤30ms @512²(中位数,≥30 次取样),报告 CPU 型号线程数与逐段耗时;320² 同口径 ≤15ms。

#### Scenario: 30ms 达标判定
- **WHEN** 在目标 Windows 机器运行引擎基准(512²,30 次取中位)
- **THEN** 端到端 ≤30ms 判定达标;输出特征构造/量化 GEMM/注意力/逐元素分段耗时;若未达标给出瓶颈段与降档建议(静态 int8、降内部边长、4x4 窗)

