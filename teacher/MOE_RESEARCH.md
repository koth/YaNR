# MoE 健壮性调研与 YaNR 适配判断

调研日期：2026-10-16。范围：原论文、作者公开训练记录、官方实现，以及本项目代码核查。
本轮没有改动模型、训练代码或 C++ 引擎，没有运行新的训练或速度基准。

## 结论与证据边界

没有足够证据把某个 MoE 宣布为所有任务上“最健壮”。需要分别考察：专家是否长期饿死、训练是否发散、输出对小扰动是否稳定、部署是否可复现，以及真实前向耗时。

对 YaNR，我建议：

- 工程参考优先选 **Megatron-Core 的 dropless Top-k 路径**，借鉴其精度、统计、状态保存和分发契约，不直接迁入整个分布式训练框架。[官方说明](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- 均衡策略采用 **DeepSeek 式动态偏置作为成熟对照，Quantile Balancing（QB）作为重点候选**。现有资料支持比较这两种方案，尚不能证明 QB 在本项目的 Top-1 图像蒸馏中更好。
- 第一轮保留现有专家数、宽度、窗口和门控公式，只比较均衡机制。共享 attention、共享专家、Top-2 应分开消融，避免无法归因。
- 长期训练覆盖与单图计算均衡分开验收。任何基于历史统计并在推理时冻结的偏置，都不能保证任意单图严格均分。

前面对“75% 分配偏离原始 argmax”的解读需要收窄：这不是错误率，也不能证明纹理问题来自路由。当前实现中目的地不依赖该窗口对欠载专家的分数，才是具体值得核查的机制。

## 候选方法比较

| 方法 | 优点与已有依据 | 关键边界 | 本项目判断 |
| --- | --- | --- | --- |
| Token Choice Top-k + 负载辅助损失 | 经典方案，工程覆盖广；选择与任务损失共同学习 | 系数需要调；均衡梯度可能与任务梯度冲突；不保证精确均衡 | 必须保留的参考方案，不应因当前代码注释就排除 |
| 动态专家偏置 / Loss-Free Balancing | 均衡反馈不通过额外辅助损失反传；选择偏置与输出权重分离 | 固定更新步长存在适应速度与震荡的取舍；小 batch、分布变化仍会不均衡 | 成熟对照，改造范围较小 [1][2] |
| QB | 从路由分数分布计算下一步偏置，减少固定步长控制器的调参；有公开大规模训练记录和官方工程实现 | 一步更新是近似，不是每批精确分配；小样本、同分、非整除必须处理 | 重点候选，不能原样复制 [3][4] |
| BASE / 最优平衡分配 | 在硬配额下优化总亲和分数，可明确约束分配数量 | 仍然强制配平；联合分配带来计算和组内依赖 | 适合检验“当前座位表策略是否损伤质量”，不是无约束路由 [5] |
| Sinkhorn / S-BASE | 可通过连续归一化近似平衡分配，已有 Megatron 路径 | 软矩阵的边际平衡不等于离散 Top-1 的计数平衡；当前官方实现训练与推理选择过程不同 | 不优先用于强调训练部署一致性的 YaNR [6] |
| Expert Choice | 每个专家选固定数量 token，专家侧计算量受控 | 每个 token 得到的专家数不固定，可能没有专家处理，仍可走残差；改变当前计算契约 | 不是当前全覆盖 Top-1 的直接替换 [7] |
| Soft MoE | 连续可微，固定专家 slot 数；视觉识别论文报告良好训练与扩展行为 | slot 是多个位置的加权混合，输出再组合；不是原位置逐 token 的稀疏 FFN | 可以研究，但图像识别证据不能直接证明纹理复刻能力，不作为首次改造 [8] |

## 为什么 QB 值得研究，但还不能称为最稳

固定步长偏置只看“过载还是欠载”。QB 使用各 token 进入 Top-k 的阈值和每个专家的路由分数分布，计算下一步偏置；推理时仍可使用冻结偏置加普通 Top-k。

Marin/Open Athena 的作者记录报告：32B 总参数、5B 激活参数模型训练 326B token，没有观察到 loss spike。这是有价值的可审查运行记录，但不是同架构、多随机种子、同计算预算的 YaNR 对比实验。该记录也明确指出，一步 QB 近似下仍有专家长期处于目标负载约 70%，不能把“无控制器步长”等同于“严格均衡”。[作者记录](https://marin.community/blog/2026/04/10/quantile-balancing/)

2026-09 的 EQB 论文进一步区分“精确计算全局分位数”和“得到精确平衡分配”。论文指出，一次同步分位数更新通常并没有解完耦合的最优化问题；其自身实验也只有一种模型规模和一个 seed，吞吐没有直接测量。因此，本项目不应把 EQB 名称中的 Exact 理解为任意输入都均分。[论文及局限](https://arxiv.org/html/2609.28053)

具体实现也不能一概称为“无超参数”：本轮查阅的 Megatron main 会对跨 microbatch/rank 的分位数估计取平均，并通过 `moe_router_quantile_balancing_ema` 平滑后更新。分位数的平均不是合并样本的精确分位数；这是工程近似，也说明必须核对版本与实际调用链。[更新代码](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/finalize_model_grads.py)

2026-09 的 ID Balancing 从控制器角度改进偏置反馈，也有更稀疏模型上的实验。但作者声明其验证范围限于一族 decoder-only 模型；不据此认定它在 8 专家 Top-1 图像网络上优于更简单方法。[论文](https://arxiv.org/abs/2609.39137)

## 官方库的选择

| 实现 | 可借鉴部分 | 对 YaNR 的限制 |
| --- | --- | --- |
| Megatron-Core | Router 精度、偏置 buffer、Top-k 与 gate 分离、dropless 分发、分层统计；main 已有 QB 路径 | 大型分布式 LLM 框架；main 功能不等于稳定 release；不能把其设备配置或性能直接套到 3080/3090 |
| MegaBlocks | 不丢 token 的稀疏/分组专家计算，避免通过丢弃溢出 token 控制成本 | 解决执行问题，不自动解决专家学习塌缩；官方训练加速比不代表本项目速度 [9] |
| Tutel | 单设备/CPU 示例、自定义 gate/expert、dropless capacity 模式，可作为独立实现对照 | 现有原生 C++ 引擎仍需要自己实现同一数值契约；README 的 Windows PyTorch 示例不是已验证 YaNR Windows GPU 集成 [10] |

工程上应移植可验证的路由语义与测试边界，而不是仅因一个库快，就把其默认容量限制、丢 token、随机第二专家或归一化门控一并带入。

## 本项目代码核查

代码位置以本次工作区为准。

### 1. 当前是整个专家块切换，影响范围大于普通 FFN MoE

[MoEBlock](D:/work/YaNR/teacher/nr_student.py:492) 和 [MoEVitBlock](D:/work/YaNR/teacher/nr_student.py:535) 不仅对 FFN 分专家，还对 QKV 和 attention 输出投影分专家。相同路由同时改变多条计算路径。

DeepSeek-V3 官方参考结构将 MoE 用于 FFN，attention 独立；其 shared expert 同样是 FFN。共享 attention 可以作为降低路由影响范围的候选，但“当前专家 attention 导致纹理问题”仍然只是待验证假设。[官方代码](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)

图像恢复领域也不是只有一种 MoE：MoCE-IR 使用共享分支、复杂度不同的专家和图像级路由，说明恢复任务存在不同设计，但其退化恢复结果不能证明对 DLSS NR 蒸馏的优劣。[CVPR 2025 原论文](https://arxiv.org/abs/2411.18466)

### 2. 当前“ST-MoE z-loss”实际是 logits L2

[当前 moe_z_loss](D:/work/YaNR/teacher/nr_student.py:666) 实现为：

```python
mean(logits ** 2)
```

ST-MoE 的定义是：

```python
mean(logsumexp(logits, dim=-1) ** 2)
```

两者不能共用相同系数并当作等价实现。逐项 L2 满足 `mean(z²) = Var(z) + mean(z)²`，直接惩罚专家之间的分差；这与约束 log-partition 不同。当前默认 `moe_z=1e-2` 不能直接与论文系数对比，也不能仅凭系数就认定它过强，需要测加权梯度与任务梯度。[ST-MoE §3.3](https://arxiv.org/html/2202.08906)

### 3. 辅助损失的注释不能作为否定 Switch/GShard 的依据

[_balanced_pick 的注释](D:/work/YaNR/teacher/nr_student.py:407) 把 `p≈均匀` 当成均衡损失恒为 1、梯度为零的原因，这个推论一般不成立。

对本项目的 `L = E * sum(f_e * mean(p_e))`，如果事后配平使 `f_e=1/E`，才有 `L=sum(mean(p_e))=1`，对 softmax logits 的梯度恒为零。相反，即使概率恰好均匀，只要实际 hard load 不均匀，仍可以有非零梯度。

本轮 CPU / float64 自动微分核查，64 个 token、8 个专家、相同全零 logits：

| 路由 | 专家计数 | 辅助损失 | 对 logits 的最大绝对梯度 |
| --- | --- | --- | --- |
| 原始 argmax | `[64,0,0,0,0,0,0,0]` | 1.0 | 0.013671875 |
| 当前 quota 后 | `[8,8,8,8,8,8,8,8]` | 1.0 | 0.0 |

这是数学机制核查，不是纹理质量的因果消融。当前辅助损失来自配平后的计数，不能据它无效就判断经典辅助损失本身无效。

### 4. Top-1 不能照搬多专家归一化

DeepSeek 官方 sigmoid gate 对选中的 Top-k 权重再归一化。若直接改成 Top-1，这一步得到 `s_selected / s_selected = 1`，通过该门控到 router 的任务梯度为零；离散 argmax 本身也没有普通梯度。

本轮 float64 自动微分确认了这一点。因此第一轮保留当前 `8 * p_selected` 契约，或另行验证不归一化 sigmoid gate；不能随意套用 Top-2/Top-8 的门控代码。

### 5. 深层路由规模与现成 QB 的前提不匹配

按 [geometry_from_valid](D:/work/YaNR/teacher/nr_geometry.py:29) 实际计算，各层 8×8 窗数量如下；数值包含本项目规定的 field padding，不能用原图边长直接除。

| 输入有效尺寸 | full | d0 | d1 | d2 | d3 | d4 | ViT token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 320×320 | 1600 | 400 | 100 | 25 | 9 | 4 | 64 |
| 512×512 | 4608 | 1152 | 288 | 72 | 20 | 6 | 96 |
| 1280×720 | 16128 | 4032 | 1008 | 252 | 66 | 18 | 288 |

查阅时 Megatron main 的 `qb_dual_update` 有 `num_tokens * k % num_experts == 0` 断言；其调用代码在 global batch 边界更新偏置，并且使用跨 microbatch/rank 的统计。不能仅复制函数后按本项目每张图、每层独立调用。

YaNR 至少需要明确：跨样本/步骤聚合、有效项 mask、非整除分位数定义、同分处理、空专家、统计何时更新、断点恢复，以及 EMA 权重对应哪份偏置。窗口边缘含不满 64 个像素的情况时，均分窗口数也不一定等于均分实际专家行数/计算量。[Megatron QB 实现](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_utils.py)

## 建议的验证顺序

下面是建议，尚未执行。

1. 先明确并固定数值方案。用户已经验证有问题的 BF16/随机路由因素，应在各比较组使用相同处理；router 应在投影前进入 FP32，仅在低精度投影之后 `.float()` 无法找回此前精度。z-loss 的修正或替换单独记录，避免与均衡策略一起变化后无法归因。
2. 路由对比保持相同专家权重结构、8 专家、Top-1、门控幅值和数据预算：当前 quota 对照、动态偏置、QB。可以补充真实 hard load 驱动的辅助损失对照。改变路由需要匹配训练；不能用旧权重直接换路由的单次出图作为结论。
3. 训练统计至少覆盖每层最大/平均负载、最小/平均负载、长期未更新专家、门控幅值、router 梯度、分差和偏置变化。平均概率熵或负载均匀不能单独证明专家学出了有用分工。
4. 图像验收以教师为目标，比较皮肤 ROI 的频段增益、与教师高频的对齐、固定噪声输入的复刻与跨噪声响应；PSNR 接近原图不是纹理蒸馏成功指标。增加固定 seed 的轻微平移/亮度扰动和连续帧检查。
5. 候选进入完整训练前应做多个 seed 的筛选；短训可以淘汰明显失败，不能证明长期不塌缩。运行预算与实际训练配置必须记录。
6. 部署使用冻结偏置；验证单图与混 batch、CPU/CUDA/Python 的路由和输出。边界同分及近同分时需一致的 tie-break，同时检查小数值差异是否放大为换专家。记录 warmup 后的前向 median 和 p95，以及路由、重排、专家 GEMM 的耗时。不能把更好的 MaxVio 当成已经加速。
7. 均衡方案确定后，再单独比较共享 attention、小共享 FFN，以及 Top-2。相同专家宽度下 Top-2 增加专家计算；改窄专家匹配 FLOPs 又会改变容量，应同时报告总参数、激活参数、实际延迟与质量。

## 主要资料

- [1] [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664)，2024。偏置选专家、原始分数加权；固定步长反馈的依据。
- [2] [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v2)，§2.1.2、§4.5.3。报告保留了极弱的 sequence-wise 辅助损失；相同 batch 均衡水平下，batch-wise 辅助损失也得到相近质量。不能简化成“所有辅助损失都不好”。
- [3] [Marin/Open Athena QB 训练记录](https://marin.community/blog/2026/04/10/quantile-balancing/)，2026-04-10。作者公开运行记录；不是本项目对比实验。
- [4] [Exact Quantile Balancing and Load-Error Injection](https://arxiv.org/abs/2609.28053)，2026-09-23。全局分位数、局部负载与保证范围的区别。
- [5] [BASE Layers](https://proceedings.mlr.press/v139/lewis21a.html)，ICML 2021。平衡约束下的线性分配。
- [6] [Megatron Router 源码](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/router.py)、[工具函数](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_utils.py)、[全局更新](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/finalize_model_grads.py)。main 为可变引用，正式移植必须固定 commit 和测试。
- [7] [Expert Choice Routing](https://arxiv.org/abs/2202.09368)，NeurIPS 2022。专家桶容量与每 token 可变计算。
- [8] [From Sparse to Soft Mixtures of Experts](https://arxiv.org/abs/2308.00951)，ICLR 2024。连续 dispatch/combine，需关注 slot 混合与本任务的空间对齐。
- [9] [MegaBlocks 官方仓库](https://github.com/databricks/megablocks)、[论文](https://arxiv.org/abs/2211.15841)。执行效率与学习均衡是不同问题。
- [10] [Tutel 官方仓库](https://github.com/microsoft/Tutel)。自定义专家/门控与 dropless 模式。

QB 原始推导的作者页面 `https://spaces.ac.cn/archives/11619` 本轮访问返回 403；其细节通过可访问的 EQB 原论文、Megatron 源码和作者训练记录交叉核查，没有把未读到的页面作为已验证内容。
