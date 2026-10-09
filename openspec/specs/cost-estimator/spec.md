# cost-estimator Specification

## Purpose
TBD - created by archiving change fast-student-distillation. Update Purpose after archive.
## Requirements
### Requirement: 逐块 MAC 成本模型
`cost_model.py` SHALL 从 manifest.json 与 nr_geometry 推导逐块 GMAC,分项覆盖 plain/expert/split/ViT FFN、qkv/proj、窗口注意力与过渡算子;teacher 形状在 512² 的总重放值 SHALL 落在 68.5 G ±3%。

#### Scenario: teacher 形状重放
- **WHEN** 以 teacher 形状运行成本模型,尺寸 512×512
- **THEN** 输出逐级 GMAC 表,总计在 68.5 G ±3% 区间,且逐级占比与 design.md 成本地图一致

#### Scenario: 分项公式校验
- **WHEN** 对单个 plain/expert/split/ViT 块按公式单独计账
- **THEN** 各分项(FFN、qkv/proj、注意力)之和与逐块输出一致,误差为 0

### Requirement: 实测校准的耗时预估
成本模型 SHALL 用 profile_nr.py 的 320²/512²/768²/1080p 实测数据校准吞吐上限,并输出毫秒预估;对 teacher 形状各分辨率的预估误差 SHALL ≤10%。

#### Scenario: 耗时预估对账
- **WHEN** 用校准后的模型预估 teacher 在 512² 的单帧耗时
- **THEN** 预估值与实测 113 ms 的偏差 ≤10%

### Requirement: 学生形状 DSL 与扫描
形状 DSL SHALL 表达逐级通道/hidden/enc+dec 块数/expert 配置/注意力策略与 ViT 配置,并对非 32 倍数通道、expert 不整除等非法输入显式报错;扫描工具 SHALL 能输出批量候选的预测加速比。

#### Scenario: 非法形状被拒绝
- **WHEN** 提交通道数不是 32 倍数的学生形状
- **THEN** 解析报错并指明非法字段,不产生任何预算输出

#### Scenario: 候选扫描
- **WHEN** 运行 sweep 脚本覆盖宽度×深度×全分辨率策略×ViT 块数网格
- **THEN** 产出 CSV,每行含逐级占比与预测加速比,可筛出 ≥5.0× 候选

