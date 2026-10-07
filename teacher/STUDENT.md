# 学生模型与蒸馏方案

> 蒸馏自 DLSS5 NR 教师(`OpenDLSS-NR` 复原,71 块 / 147MB / 512² 113ms)的小模型。
> 本文是架构与训练的权威参考;决策依据见
> [openspec/changes/fast-student-distillation/design.md](../openspec/changes/fast-student-distillation/design.md)(D1–D11)。
> 实测基准(3090):**512² 11.05ms = 10.2× 教师**,9.29M 参数(教师 ~147M 等效)。

---

## 0. 一页概览

| | 教师 | 学生(student_v0) |
|---|---|---|
| 结构 | 71 块:6 级 UNet + 8 块全局 ViT | **19 块**:同拓扑 6 级 UNet + 3 块 ViT,全分辨率块去注意力 |
| 权重 | 147MB(fp8 打包) | 9.29M 参数(fp32 约 37MB / f16 约 18MB) |
| 数值 | fp8(E4M3)定点链、16 乘积组、f16 累加 | **标准 matmul + 常规 softmax**(数值自由) |
| 训练 | —(NVIDIA 权重) | bf16 AMP + f32 主权重,**无 QAT** |
| 推理 | 自研 fp8 CUDA 链 | torch.compile(reduce-overhead)+ autocast f16 |
| 512² 耗时 | 113 ms | **11.05 ms(10.2×)** |
| 512² MACs | 69.6 G | 13.7 G(5.08×,这是保底;后端吞吐差再放大一倍) |

---

## 1. 与教师相同的契约(不能动)

学生的输入输出与合成语义和教师逐位同构 —— 这是它能直接替换进管线、且蒸馏可对齐的前提。

**输入:16 lane 特征**(f32 `[full_rows, 16]`,填充场镜像 `(2·valid − x − 2)`,噪声按填充坐标哈希):

| lane | 内容 |
|---|---|
| 0-2 | 高斯噪声(Box-Muller,hash(坐标, seed),师生同 seed) |
| 3 | 恒 1 |
| 4-6 | proxy 三半舍入居中(`roundF16(roundF16(roundF16(c)−0.5)·0.125)`) |
| 7-9 | **上帧输出**同样居中(D10:存储=blend 后 composite 码值、`truncate_half` 向零截断、5-tap Catmull-Rom 重投影;无历史时 = lanes 4-6 的拷贝) |
| 10-15 | style/128、tone、structure 三元组(automask 语义) |

**输出:head f32 `[full_rows, 4]`**:

```
neural = clamp(proxy + head[0:3]/4, 0, 1)          # rgb 残差,1/4 幅度注入
weight = clamp(sigmoid(head[3]) * blend_scale, 0, 1)  # blend_scale = 模型常量(教师 0.7397)
neural = neural + (history - neural) * weight        # 时序合成(仅有历史时)
```

**几何**([nr_geometry.py](nr_geometry.py),逐位不动):field/padding 规则、六级池化尺寸、
8×8 窗口相位循环 `PHASES = [(0,0),(4,4),(4,0),(0,4)]`、ViT token 数与 padding。

---

## 2. 拓扑总览

六级编码/解码 + 底部 ViT。每级 `enc = blocks//2`、`dec = 其余`(full 级 1+1)。
下表为 `shapes/student_v0.json` 在 512² 的实算(cost_model.py):

| 级 | rows(512²) | 通道 | hidden | 块(enc+dec) | 类型 | 注意力 | GMAC | 占比 |
|---|---|---|---|---|---|---|---|---|
| full | 294912 | 32 | 32 | 1+1 | FFN | **无** | 3.62 | 26.5% |
| d0 | 73728 | 32 | 32 | 2+2 | FFN | 8×8 | 3.02 | 22.1% |
| d1 | 18432 | 64 | 64 | 2+2 | expert | 8×8 | 3.02 | 22.1% |
| d2 | 4608 | 128 | 64 | 1+1 | expert | 8×8 | 1.28 | 9.4% |
| d3 | 1152 | 256 | 64 | 1+1 | expert | 8×8 | 1.17 | 8.5% |
| d4 | 320 | 256 | 64 | 1+1 | split(E8, bc32, mc128) | 8×8 | 0.29 | 2.1% |
| ViT | 96 tokens | 512 | FFN 1024 | 3 | ViT | 全 token | 0.64 | 4.7% |
| 过渡/适配/head | — | — | — | — | gemm | — | 0.63 | 4.6% |
| **合计** | | | | | | | **13.71** | **5.08×** |

数据流:

```
16 lane features
  └─ adapter(16→32)────────────────────────────────────────────┐
      └ [full enc 1块: FFN-only] ── skip_full ─────────────┐   │
          └ 2×2 pool + linear(32→32)                        │   │
              └ [d0 enc 2块] ─ skip_d0                       │   │
                  └ pool + linear(32→64)                     │   │
                      └ [d1 enc 2块] ─ skip_d1               │   │
                          └ … [d2 enc 1块] … [d3 enc 1块]    │   │
                              └ [d4 enc 1块] ─ skip_d4       │   │
                                  └ pool→96tok + linear(256→512)
                                      └ [ViT ×3(全 token 注意力)]
                                          └ linear(512→256)
  解码(每级:linear(高→本级通道)→ 最近邻 2× 上采样 + skip·aux → dec 块):
      d4 → d3 → d2 → d1 → d0                                  │
          └ post_blend(d0 上采样·aux0 + skip_full·aux1) ──────┘
              └ [full dec 1块: FFN-only](教师 block70 的角色)
                  └ head(32→4) → rgb 残差 + blend logit
```

---

## 3. 块设计(语义与教师同构,数值自由)

所有块共享:**残差 + 每通道 aux 缩放**(`out = f(x) + x * aux`,`aux` 可学习、初值 1),
这保留了教师"带缩放跳连"的函数族,逐级特征对齐更容易。

| 块 | 结构 | 对应教师 |
|---|---|---|
| `FFNBlock` | `y = contract(SiLU(expand(x))) + x·aux_ffn`;可选注意力尾 | 教师标准块的自由数值版 |
| `FFNBlock(attn=False)` | 同上去掉 qkv/注意力/proj(**full 级专用**,D3) | block 0 / 70 降级 |
| `ExpertBlock` | E 个全输入→h 展开、h→32 窄支、合并(E·32=ch)→ ch→ch 契约 | 教师 expert FFN(窄瓶颈 32 同款) |
| `SplitBlock` | branch(ch→ch)→ E 个 bc→mc→bc 窄支 → contract;**E·bc = ch 分组切片** | 教师 512 级 split 块 |
| `VitBlock` | expand→contract(残差)→ qkv → 全 token 注意力 → proj(残差) | 教师 ViT 块 |

注意力尾 `_AttentionTail`:`qkv(ch→3ch) → WindowAttention → proj(ch→ch) + x·aux_attn`。

**相位在构造期定死**:教师的窗口相位是"每级计数器按块序号走 `PHASES[i & 3]`"(enc 块 0..n-1、
dec 块续计),学生把它在 `__init__` 里定进每块的 `shift_x/shift_y`。这既是语义事实,也消掉了
torch.compile 的动态整数问题(见 §6 事故记录)。

**规格约束**(DSL 校验强制):通道/hidden 为 32 的倍数;split 需 `branches × branch_channels = channels`
(教师的分组切片语义);window attention `heads = ch/32`。

---

## 4. 注意力

**`WindowAttention`**(8×8 移位窗,可 16 槽):
- q/k 余弦归一化(`x/‖x‖`),q 乘可学习每头 scale;
- 可学习相对先验 `prior [heads, slots, slots]`(教师 relativeBias 的角色,初值 0);
- 窗口索引:窗口网格交叉积 × 槽位,越界槽 valid 掩码 + 值清零(gather 语义同教师参考路径);
- **NaN 安全掩码 softmax**:`finfo.min` 替代 `-inf` 做掩码、amax 平移、`exp·mask / Σ`(事故见 §6)。

**`VitAttention`**(底部全局):q 带 `√32 · scale`、列掩码到 tokens、常规 softmax;
padding 列不参与(教师的 vitExpWeight 校正被"掩码 softmax"这个标准做法取代)。

与教师的数值差异(D11):无 expWeight 查表、无 63 节点 softmax 树、无 f16 舍入链 ——
教师那套是复现 NVIDIA 网络的约束,不是速度来源;学生用常规算子 + 后端优化(§6)。

---

## 5. plumbing(语义照抄教师)

| 算子 | 语义 |
|---|---|
| `box_downsample` | 2×2 box:`((a+b)+(c+d))·0.25`,越界位 0(教师 downsample 同款) |
| `UpsampleMerge` | 最近邻 2×(src>>1,clamp)+ 编码 skip·`aux`(教师 upsample_residual) |
| `PostBlend` | d0 解码输出上采样·`aux_pair[0]` + full 跳连·`aux_pair[1]`(教师 block70 前的合并) |
| `head` | 32→4 线性,输出 f32 |
| `blend_scale` | 可学习标量(教师为模型常量 0.7397) |

---

## 6. 数值与推理后端(D11:怎么快怎么来)

**学生不继承 fp8 定点链。** 关键事实:教师 fp8 链实测吞吐 ~610 GMAC/s ≈ 3090 f16 张量核峰值的
2%;换标准 f16 matmul + 常规 softmax 后有数量级空间。连带解锁:无 QAT、通道/hidden/窗口自由、
任意融合策略。

训练:bf16 AMP + f32 主权重;推理:autocast f16。训练/推理 dtype 差距的质量守门在 eval 阶段做
(3.5,目标 ≤0.1dB)。

**基准(bench_student.py,3090,随机权重,20 次取均值)**:

| 分辨率 | eager f32 | eager f16 | compile f32 | **compile f16** | 教师 | 加速 |
|---|---|---|---|---|---|---|
| 320² | 21.5ms | 23.5ms | 10.1ms | **9.69ms** | 57ms | **5.9×** |
| 512² | 26.6ms | 24.1ms | 13.2ms | **11.05ms** | 113ms | **10.2×** |
| 768² | 42.4ms | 29.9ms | 23.1ms | **14.18ms** | 231ms | **16.3×** |

- 胜者:`torch.compile(mode='reduce-overhead')` + autocast f16;有效吞吐 1241 GMAC/s(512²)、
  2087(768²)= 教师链的 2–3.4×。
- 零手写 kernel 即达成;手写融合(任务 3.3)为可选优化。
- MACs 记账的 5.08× 是保底(按教师吞吐折算);实际加速 = MACs 比 × 后端吞吐比。

**CPU 引擎(`engine/`,fp32 + AVX2,M3)**:i9-10850K(10C/20T,无 VNNI)、`OMP_NUM_THREADS=8`
(20 线程 HT 过订阅反而慢 ~8×)、机器上有训练占 ~2 核(数字有 ±10% 噪声);30 次 median:

| 分辨率 | 朴素基线 | M3 现状 | 教师(同机) | 30ms 口径 | 累计 |
|---|---|---|---|---|---|
| 320² | 237ms | **38.5–41ms**(min 34.5) | — | 差 ~15–20% | **~5.9×** |
| 512² | 767ms | **120–127ms**(min 111) | ~3.2s | 物理不可达 | **~6.1×** |

剖析(320²):gemm ~25ms / attn ~8ms / other ~6ms / silu 0。attn 内部:av 2.4 /
scores 2.6 / norms 0.7 / softmax 0.5(墙钟)。M3 手段:打包 [K][N] GEMM + 4×16
AVX2 微内核(14 ymm 零溢出)、自适应 MC 分块、帧间 arena、向量化 exp、silu 与残差
出口融合、upsample+跳连一趟融合、qkv 按窗口槽序写输出(gemkn_rows,排列缓存,
gather 归零)、attention 核心双查询×向量化(k 转置打分 + 共享 v 加载的 av)。
与 torch 逐级对账 max|diff| ≈ 4e-7(120dB)。

**到 30ms 的账**:512² 硬地板 ~55ms(GEMM 打满也顶不掉 attention/元素级),
320² 现 38.5(min 34.5),纯工程侧收敛 ~35-36;再往下要么砍 ~10-15% 形状
(重训,质量小损),要么换硬件口径。**测量注意**:v1 训练并行时中位数 ±10% 噪声,
终数要等训练结束/空载机重测。

**int8(任务 8.2,NR_INT8=1 混合路径)**:per-row 动态 A 量化 + per-channel 权重量化,
i16 展开 vpmaddwd 安全内核(无饱和);按 K 阈派发。`gemm_bench` 全形状微基准结论:
K≥256 i8 快 2–3×、K=128 平、K≤64 慢 3–5×(hsum/量化开销)⇒ 阈值 K≥256。质量
max|diff| 8.5e-4 **过 2e-3 parity 线**(composite 等效 84.8dB)。但网内实测与 fp32 平手
(冷缓存/带宽争抢吃掉微基准收益);**这台无 VNNI 的机器上 int8 不是杠杆**,它的价值在
VNNI 硬件(8.7 验收机):vdpbusd 直接 u8s8→i32 累加,预计 2–4×,那才是 512²/30ms 的路。

到 30ms 的差距(硬算术):512² 13.7 GMAC ÷ 30ms = 457 GMAC/s 只是 GEMM 一项,而这台
10 核 AVX2 fp32 峰值 ~600 GMAC/s —— 即便 GEMM 打满,attention+元素级的地板也把总量
顶在 ~55ms。**30ms@512² 在这台硅上物理不可达**;320² 的 30ms 差 ~15-20%,收敛后
~35-36ms,余量靠形状折扣(~10-15%,重训)或空载机复测挤。

---

## 7. 成本记账(cost_model.py)

每像素每块 MACs(64 槽注意力 = `128·ch`):

```
plain  FFN 块:  2·ch·h + 4·ch² + 128·ch
expert FFN 块:  ch²·(h/32 + 1) + ch·h + 4·ch² + 128·ch      # E = ch/32,窄瓶颈 32
split 块:       6·ch² + 2·E·bc·mc + 128·ch                   # branch/contract 2ch² + qkv/proj 4ch²
ViT 块:         2·ch·ffn + 4·ch²(注意力另计 2·T·Tpad·32·heads)
过渡 gemm:      rows · k · n(enc ch→2ch、dec 2ch→ch、ViT 进出)
```

三候选(冻结于任务 1.9,均过 ≥5× 门禁):

| 形状 | 512² GMAC | 加速 | 定位 |
|---|---|---|---|
| `shapes/student_v0.json` | 13.71 | 5.08× | **默认**:深层层级保宽度砍深度 |
| `shapes/student_alt_depth.json` | 13.16 | 5.29× | 深层欠拟合时换 |
| `shapes/student_alt_attn4x4.json` | 12.80 | 5.43× | full 级 4×4 窗注意力备选(需 16 槽 kernel) |

用法:`python3 cost_model.py --shape shapes/student_v0.json --sizes 320,512,768,1080 [--json]`。

---

## 8. 蒸馏方案

### 8.1 在线教师(D7)

每 step 现跑教师([annotate.py](annotate.py) `TeacherAnnotator`,融合 CUDA 路径,512² 一帧 113ms),
**不缓存中间激活**(full-res 捕获点 18MB/点,5 万样本破 TB)。可缓存的只有小输出(逐帧 head f32、
stored history f16)。step 预算 ≈ 教师 113ms + 学生 fwd/bwd ≈ 250–950ms(320²)。

### 8.2 特征对齐表(student_alignment,13 点)

| 学生捕获点 | 教师捕获点 | 形状(512², s/t) | 处理 |
|---|---|---|---|
| s-enc-full | block-0 | [294912, 32/32] | 直接对齐 |
| s-enc-d0 | block-4 | [73728, 32/32] | 直接对齐 |
| s-enc-d1 | block-8 | [18432, 64/64] | 直接对齐 |
| s-enc-d2 | block-14 | [4608, 128/128] | 直接对齐 |
| s-enc-d3 | block-22 | [1152, 256/256] | 直接对齐 |
| s-enc-d4 | block-30 | [320, 256/512] | 1×1 投影(t→s) |
| s-vit | block-38 | [96, 512/1024] | 1×1 投影 |
| s-dec-d4 | block-47 | [320, 256/512] | 1×1 投影 |
| s-dec-d3..d0 | block-55/61/65/69 | 同通道 | 直接对齐 |
| s-head | head | [rows, 4] | 输出损失 |

投影层(`FeatureAligner`,无偏置)随训练走,部署时丢弃。

### 8.3 损失(train_distill.py)

```
L = 1.0·L_out + 0.5·L_detail + 0.5·L_feature(+ 0.0·L_temporal,run v1 低权启用)
```

| 分量 | 定义 | 备注 |
|---|---|---|
| L_out | `L1(rgb/4) + 0.5·SmoothL1(blend logit) + L1(neural)` | 门禁量的就是合成图 PSNR |
| L_detail | 拉普拉斯 2 级 + Sobel **场之差** L1 | 初版误写"两图能量之和"(奖励模糊),已修,+1.8dB |
| L_feature | 逐点 `(1−cos) + 0.1·MSE`,按点均值 | 初版 MSE 除以 `t.var()`(方差小的层爆 1e4 → NaN),已修 |
| L_temporal | 相邻帧输出差分 vs 教师差分一致 | 序列模式;历史 = 教师 rollout(D10) |

### 8.4 训练配置

AdamW(lr 2e-4, wd 0.01)、warmup + cosine、grad clip 1.0、**EMA 暖起** `decay = min(0.999, (1+s)/(100+s))`
(固定 0.999 在短跑会被半成品权重污染:32% 质量来自 step1000–2000 → EMA 掉到 24dB,已修)。
lane 逐样本采样(style 0–3、tone 0.3–0.7、structure 0.2–0.8);噪声 seed 师生一致、确定性。
日志:csv + **CometML**(project `dlss-student`,逐 step 损失分量/逐点特征/PSNR/图)。

### 8.5 时序蒸馏(D10 teacher-forcing)

历史链 = 教师 rollout 的 composite(`truncate_half` 存储)+ 解析/给定运动的 5-tap Catmull-Rom
重投影;师生吃同一历史起步,学生 rollout 微调放后期。首帧 lanes 7-9 = 当前 proxy。

### 8.6 已验证(过拟合门禁,任务 5.11)

8 样本过拟合:裸权重均值 **PSNR 40.50 dB ≥ 40**(门禁通过),三轮修复史:warmup 配置 →
detail 场之差 → EMA 暖起。管线正确性由此背书。

---

## 9. 训练数据

| 来源 | 内容 | 量 | 状态 |
|---|---|---|---|
| DIV2K_train_HR(ETH) | 2K 干净图 | 800 张 | ✓ 下载入账 |
| DAVIS-2017 Full-Res | 视频帧序列(时序用) | 60+ 序列 | ✓ 下载入账 |
| 程序化([gen_sequences.py](gen_sequences.py)) | 6 类场景(平移/缩放/旋转/粒子/棋盘混叠/流场) | 200 序列 × 8 帧 | ✓ 已生成 |
| [degrade.py](degrade.py) | proxy 退化链(混叠→糊化→光晕→噪声→色带→残影) | 按需 | ✓ |

数据装配注意(任务 4.10):DAVIS 的 480p/全分辨率重复、Annotations 是掩码图,递归收集会污染,
run v1 用清单式白名单。

---

## 10. 现状与路线

| 阶段 | 状态 |
|---|---|
| 成本模型 / 形状冻结(1.x) | ✅ |
| 学生骨架 / 基准(2.x、3.1-3.2) | ✅ 512² 11.05ms = 10.2× |
| 数据线(4.1-4.5) | ✅ 下载/生成/退化/教师标注 |
| 训练管线 + 过拟合门禁(5.1-5.11) | ✅ 40.50dB |
| 5.12 sanity run(800 张、320²、20K 步) | ✅ 泛化(DAVIS 未见帧)**41.97dB** / EMA 41.96dB |
| 5.13 run v1 全量(多源数据 + 时序) | 启动中 |
| 6.x 评估验收 / 7.x 打包 | 待办 |

---

---

## 11. 术语表

| 术语 | 含义 |
|---|---|
| **lane** | 输入特征的"通道面":每像素 16 个标量 = 16 张并排的图像面(NVIDIA 文档用语)。lane 恒为 16;网络内部的 **channel**(32/64/…)随形状变,两者刻意区分 |
| **proxy** | 输入的低质量渲染(码值 0..1);lane 4-6 就是它居中后的三个面 |
| **history lane** | lane 7-9:上帧输出(合成码值,`truncate_half` 存储、5-tap Catmull-Rom 重投影后居中);无历史时 = 当前 proxy |
| **码值(code value)** | 0..1 的显示代理空间;网络残差在码值空间按 1/4 幅度注入 |
| **EMA 暖起** | 影子权重的 decay 随步数从 ~0.01 渐升到 0.999:短跑不被早期半成品污染(否则 32% 质量来自半成品 → 影子权重掉 16dB),长跑自动退化为常数 |
| **过拟合门禁** | 5.11:8 样本训到合成图 PSNR ≥40dB 才允许全量训练 —— 验证管线(对齐/损失/梯度)正确,不验证泛化 |
| **在线教师** | 训练每 step 现跑教师取目标(D7),不缓存中间激活;只缓存小输出(head/history) |
| **capture 点** | 块输出的捕获钩子(教师 79 点、学生 13 点),蒸馏的逐级特征对齐就发生在对齐表所列的点上 |
| **场之差损失** | 对师生输出的拉普拉斯/Sobel *场*取差的 L1(匹配高频形态);区别于"各自能量之和"(那是罚高频、奖励模糊的错误写法) |
| **GMAC / GMAC/s** | 十亿次乘加 / 每秒吞吐;512² 学生 13.7 G MACs,后端实测 1241 GMAC/s |

## 12. 文件索引

| 文件 | 职责 |
|---|---|
| [nr_student.py](nr_student.py) | 学生网络(块/plumbing/对齐表/capture) |
| [run_student.py](run_student.py) | 推理入口(run_image 同参,支持 `--prev`/`--motion` 时序) |
| [train_distill.py](train_distill.py) | 蒸馏训练(在线教师 + 四类损失 + EMA + Comet) |
| [annotate.py](annotate.py) | 教师标注接口(`TeacherAnnotator`、`--probe`/`--rollout`) |
| [cost_model.py](cost_model.py) / [sweep_shapes.py](sweep_shapes.py) | MACs 记账 / 形状扫描 |
| [bench_student.py](bench_student.py) | 后端基准(eager/compile × f32/f16) |
| [nr_history.py](nr_history.py) | 时序历史(truncate_half、5-tap Catmull-Rom 重投影) |
| [degrade.py](degrade.py) / [gen_sequences.py](gen_sequences.py) | proxy 退化 / 程序化序列 |
| [run_image.py](run_image.py) | 教师图像入口(16 lane 构造的权威实现) |
| [../shapes/*.json](../shapes) | 学生形状 DSL(v0 / alt_depth / alt_attn4x4) |
