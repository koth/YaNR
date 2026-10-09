## Context

老师网络(`OpenDLSS-NR` 的复原 + [teacher/](teacher) 自包含实现)是 71 块的六级 UNet + 底部全局 ViT:full(32ch)→ d0(32)→ d1(64)→ d2(128)→ d3(256)→ d4(512 split)→ ViT(1024/FFN4096, 8 块)。512² 单帧 ≈ 69 G MACs,实测 113 ms(≈610 G products/s,贴近 3090 fp8 张量核上限),权重 147 MB 每帧全量回读但只占 <7% 带宽 —— **是纯计算瓶颈,5× 必须来自 MAC 减少**。

逐级成本(512²,按 manifest 参数量 × level rows 记账,已与实测对齐 ±3%):

| 级 | rows | 块 | 通道/FFN | GMAC | 占比 |
|---|---|---|---|---|---|
| full | 294912 | 0, 70 | 32 / h128 | ~9.7 | 14% |
| d0 | 73728 | 1-4, 66-69 | 32 / h128 | ~9.7 | 14% |
| d1 | 18432 | 5-8, 62-65 | 64 / expert | ~7.9 | 12% |
| d2 | 4608 | 9-14, 56-61 | 128 / expert | ~10.0 | 14% |
| d3 | 1152 | 15-22, 48-55 | 256 / expert | ~12.1 | 18% |
| d4 | 320 | 23-30, 39-47 | 512 / 8 分支 split | ~9.7 | 14% |
| ViT | 96 tokens | 31-38 | 1024 / FFN4096 | ~9.8 | 14% |

成本在 7 个级上几乎完全平坦(每块 MACs/px ≈ FFN + 4ch² + 128ch 窗口注意力),所以学生必须**每一级都砍**,不存在"砍 ViT 就够了"的捷径。输出残差仅 ±0.03(8-bit 空间 ±7/255),blend sigmoid 均值 ≈ 0.5 —— 网络做的是温和的细节精修,这给了小模型容量空间。

## Goals / Non-Goals

**Goals:**

- 学生网络在 3090 上 512² 实测 ≥4.5×(目标 ≥5.0×)、320²/768²/1080p 同步 ≥4.5×;预测 MAC ≥5.0×。
- 质量:合成图对学生-教师 PSNR ≥ 35 dB(512²);学生-输入 PSNR 不低于教师基线(38.53 dB)1 dB;blend logit MAE ≤ 0.15;时序闪烁指标 ≤ 1.5× 教师。
- 学生保持 teacher 的几何规则、16 lane 输入、4 通道 head 与 composite 语义,可直接替换进 run_image.py 流程。
- 全部算力/形状决策由 cost-estimator 记账,预测 vs 实测误差 ≤15%。
- 训练纯蒸馏:不需要真实游戏配对数据,教师标注即可。

**Non-Goals:**

- 不改 teacher 的数值行为、权重格式与校验语义;不做 teacher 的再优化。
- 不做真路由 MoE(Plan B)、不做低秩分解、不做 2:4 结构化稀疏(留作后续,见 Risks)。
- 不追求与教师 bit-exact 或同分布噪声实现;学生是蒸馏近似,不是数值复刻。
- 不做 WebGPU/vit_fused 新 kernel 谱系的移植;不做多卡训练。
- 本期不发布任何权重(许可约束),只交付训练/评估/打包能力。

## Decisions

**D1 契约不变,只缩容量。** 几何([teacher/nr_geometry.py](teacher/nr_geometry.py) 的 field/padding/窗口相位)、16 lane 输入语义、head 32→4、composite(`clamp(proxy + rgb/4)` 与 `sigmoid(blend)`)一律不动。理由:几何决定哪些 token 存在,改它等于改任务;输入输出不变才能量化端到端质量。

**D2 学生 v0 形状:同金字塔、变窄、减深、全分辨率块去注意力。** 初始冻结点(sweep 后允许 ±20% GMAC 内调整,task 2.x 负责):

| 级 | rows | 通道 | hidden | 块(enc+dec) | 注意力 | GMAC | vs 教师 |
|---|---|---|---|---|---|---|---|
| full | 294912 | 32 | 32 | 1+1 | 无(FFN-only) | 3.63 | ÷2.7 |
| d0 | 73728 | 32 | 32 | 2+2 | 8×8 | 3.02 | ÷3.2 |
| d1 | 18432 | 64 | 64(expert E2) | 2+2 | 8×8 | 3.02 | ÷2.6 |
| d2 | 4608 | 128 | 64(expert E4) | 1+1 | 8×8 | 1.29 | ÷7.7 |
| d3 | 1152 | 256 | 64(expert E8) | 1+1 | 8×8 | 1.17 | ÷10.3 |
| d4 | 320 | 256 split(E4, bc32, mc128) | — | 1+1 | 8×8 | 0.29 | ÷34 |
| ViT | 96 tokens | 512 | FFN 1024 | 3 | ✓ | 0.64 | ÷15 |
| 过渡/适配/head | — | — | — | — | — | ~0.6 | — |

合计 ≈ **13.7 G MACs → 预测 5.1×**(cost_model.py 实算:512² 13.69 G,320² 5.67×);参数量 147 MB → ≈18 MB。交换率表:d0 4→2 块 −1.5 G;ViT 3→2 块 −0.2 G;d2/d3 各 +1 块 +0.6/0.6 G。质量不足先加 d2/d3 深度,速度不足先减 d0 深度。

**D3 全分辨率块降级为 FFN-only 细节块。** 教师在 295K rows 上跑 2 个带 8×8 窗口注意力的块,注意力项 128·ch MACs/px 乘上巨大 rows 是成本大头;学生 full 级只保留 FFN(2·ch·h + 4ch²),感受野由 d0 层注意力与 UNet 供给。理由:输出是 ±0.03 的高频残差,先验上更依赖局部非线性而非 8×8 上下文。备选(质量不达标时启用):4×4 窗口注意力(16 槽,需要 expWeight 表与 softmax 树的 15 节点版本,task 4.7)。

**D4 ViT 最狠地砍。** 96 个 token 喂 8×1024/4096 的块是明显的容量过剩(占全模型 66% 权重却只做 14% 计算);学生 ViT = 512 通道 / FFN 1024 / 3 块。权重回读从 98 MB 降到 ~7 MB,顺带利好低分辨率帧的固定开销。

**D5 复用专家 FFN 与 split 的 kernel 语义。** d1/d2/d3 保留 expert FFN(窄瓶颈 32,专家数 = ch/32),d4 split 8 分支 → 4 分支(64→256→64 改 32→128→32)。全部走 [teacher/nr_kernels.cu](teacher/nr_kernels.cu) 的 block_ffn / block_ffn_expert 路径,只做参数化扩展,不引入新 kernel 谱系。

**D6 蒸馏 = 逐级特征对齐 + 四类损失。** 损失 = 输出 L1(rgb/4 残差)+ blend logit SmoothL1 + 细节(拉普拉斯/Sobel 高频)L1 + 逐级特征 cosine/L2 + 时序差分一致;特征对齐发生在学生真实推理数值(见 D11)的激活上。学生训练**不需要 QAT**——它不活在 fp8 里。

**D7 数据 = 任意高清图/序列 + 教师标注。** 纯蒸馏不需要真实游戏配对:DIV2K/视频帧 + 程序化序列(平移/缩放/粒子/棋盘),经 proxy 退化(模糊+降采样+轻噪声)后喂教师,产出教师的 head 与蒸馏点激活。噪声 lane 师生同 seed(映射是确定的,同输入同目标)。**标注采用在线教师**:教师走融合 kernel(512² 一帧 113ms)每 step 现算,不缓存中间激活(full-res 捕获点 18MB/点,5 万样本破 TB);只存源图 crop + seed,16 lane 输入由 seed 确定性重建。step 预算 ≈ 教师 113ms + 学生 fwd/bwd ~100ms ≈ 250ms,20 万步 ≈ 14h,在 72h 内。

**D8 验收以实测为准,预测为门禁。** 形状阶段要求预测 ≥5.0×;训练完成后 512² 实测 ≥4.5× 算达标、≥5.0× 算达成目标。理由:学生的小 GEMM(64/128 通道)效率 530–710 G/s 低于大块,预测 5.1× 实测可能落在 4.5–5.0×。

**D9 先 torch 后选型。** 学生前向、训练、评估全在 torch 起步;推理后端( torch.compile / 手写融合 kernel / cutlass)以实测定(任务 3.x),不预设。训练数据与训练都在服务器,本地只做 kernel/数值/成本工作。

**D11 学生数值自由(用户决策:可以怎么快怎么来)。** 学生**不继承 fp8 定点链**:教师那套数值语义(16 乘积组、f16 累加、逐位复现)是复原 NVIDIA 网络的枷锁而非快的来源——实测 fp8 链吞吐仅 ~610 GMAC/s,约为 3090 f16 张量核峰值(35 TMAC/s)的 2%,同预算 MACs 下换标准 f16 矩阵乘 + 常规 softmax 有数量级空间。学生用标准 f16/bf16 matmul、常规 softmax、可学习余弦注意力先验,通道/hidden/窗口不再受 kernel 伪约束(32 倍数等),训练无需 QAT。MACs 记账不变:student_v0 的 5.08× 从预测值降级为**保底下限**,现实预期 8–16×。影响:第 3 组改为推理后端选型;教师 gemm 原语不复用,但块语义(残差+aux 缩放、SiLU、余弦窗注意力、post_blend)保持,逐级特征对齐不受影响。**实测验证(3090,bench_student.py)**:torch.compile + autocast f16 达 512² 11.05ms(**10.2×**)、768² 14.18ms(**16.3×**)、320² 9.69ms(5.9×),有效吞吐 1241 GMAC/s(教师 fp8 链的 2×),零手写 kernel;MACs 记账的 5.08× 确实只是保底。

**D10 时序历史语义(spike 结论,任务 4.4)。** lanes 7-9 是**上帧输出**,不是上帧 proxy:存储 = composite 码值(`neural = clamp(proxy_code + head_rgb/4)` 经 temporal blend,blend 权重 = `clamp(sigmoid(head3) × blend_scale, 0, 1)`),**不含** style/tone/显示变换;存储时 `truncate_half` **向零截断**到 f16 半格(四舍五入会逐帧单向漂移);下一帧在运动重投影位置用 **5-tap Catmull-Rom**(双线性折叠技巧,clamp-to-edge;普通双线性会逐帧融化细节)采样,再走与 proxy 相同的 `centre()` 进 lanes 7-9。无历史(首帧/无效运动)时 lanes 7-9 = 当前 proxy 的拷贝,composite 不混合。证据:[docs/network.md](OpenDLSS-NR/docs/network.md) 的 lane 表、[docs/frame.md](OpenDLSS-NR/docs/frame.md) "History reconstruction"、[frame.wgsl](OpenDLSS-NR/ports/browser-webgpu/shaders/frame.wgsl) 的 `input_features`/`compose`/`truncate_half`。训练含义:teacher-forcing 起步(师生共用同一历史 = 教师 rollout),学生 rollout 微调放 5.6;程序化序列带解析运动场可做真实重投影,静态图用恒等重投影;实现于 `teacher/nr_history.py`。

## Risks / Trade-offs

- **全分辨率去注意力丢细节风险**(D3):diff×12 视觉检查 + 细节损失兜底;若质量不达标,启用 4×4 窗口注意力备选(kernel 表工作量约 1-2 天),成本回到 +1.2 G 仍保 ≥4.5×。
- **history lane 语义未确认**(任务 3.4 spike):单帧时 history=当前 proxy,序列训练时"上帧 proxy 还是上帧输出"未定;教错会把时序特性学崩。spike 未结论前,训练 v1 以单帧为主、时序损失降权。
- **教师是生成式网络**:同一输入下教师输出含噪声驱动的随机细节,学生 L1 学成条件均值会发糊。缓解:师生同噪声输入、高频损失、必要时对残差做相关性约束(不做 GAN,见 Non-Goals)。
- **小 GEMM 效率低于记账假设**:预测 5.1× 实测可能只有 4.5×;预算留 15% 余量,验收线按实测 4.5× 设。
- **单卡 3090 训练预算**:run v1 ≤72h 是硬约束;若数据标注或吞吐超预算,先降样本量到 30K 保训练闭环,质量不足再扩。
- **深度砍太狠**(d2/d3 从 12/16 块到 2 块):容量悬崖风险,交换率表(D2)明确"质量不足买深度"的顺序,消融(6.4)出证据后调整。
