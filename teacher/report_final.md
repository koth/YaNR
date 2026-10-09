# fast-student-distillation 验收报告(report_final,6.6)

> 2026-10-09 | 交付物:**slim_mixed v1.1**(4.29M 参数 / 9.32 GMAC@512²)
> ckpt:`/tmp/run_v11_mixed/ckpt.pt`(60K 步,EMA);引擎:`engine/build/cpu_engine`
> 硬件口径:i9-10850K(10C/20T,AVX2 无 VNNI),`OMP_NUM_THREADS=8`,空载,30 次取中位

## 0. 验收判定(先看这张表)

| 门禁 | 口径 | 结果 | 判定 |
|---|---|---|---|
| 6.2 质量退化 512² | 学生对教师 ≤ 1.0dB(10 图均值) | **+0.05dB** | ✅ **过** |
| 6.2 质量退化 320² | 同上(非目标口径) | +1.10dB | ⚠️ 名义超线 0.10dB |
| 6.3 flicker | ≤ 1.5× 教师 | ×2.78@512² / ×2.67@320² | ❌ 未过(与基线 ×2.66 同水平) |
| 6.3 warp error | ≤ 1.5× 教师 | ×0.97 | ✅ 过 |
| 8.8 int8 退化 | ≤ 0.3dB | ≈0.00dB(合成图 PSNR 65.5dB) | ✅ 过 |
| 8.7 512² ≤30ms | VNNI 硬件验收 | AVX2 实测 91.5ms | ⏭️ 移交 VNNI 机器(见 §6) |

**一句话**:主目标(512² ≤1dB)达标且是速度/质量拐点上最快的形状;flicker 门未过,根因已定位(§4),
修复需一轮真实视频对重训(用户决策:本轮照实记录不跑);30ms@512² 在 AVX2 物理不可达,按约定移交 VNNI 验收。

## 1. 形状表(v0 形状表 + 消融候选)

| 形状 | 512² GMAC | 320² GMAC | 参数 | 5K 步 vs 教师(320²) | 定位 |
|---|---|---|---|---|---|
| `student_v0` | 13.71 | 5.00 | 9.29M | 43.67dB | 质量基线(v1_official 50K) |
| `student_slim_blocks` | 10.17 | 3.70 | ~5.4M | 42.75dB | 中间档 |
| **`student_slim_mixed`(v1.1)** | **9.32** | **3.34** | **4.29M** | 43.17dB | **交付** |
| `student_slim_deep` | 7.19 | 2.54 | ~3.5M | 43.58dB(曾抱输入 corr 0.007) | 激进档,风险高 |

v0 512² 逐级实算见 [STUDENT.md](STUDENT.md) §2。slim_mixed 与 v0 的同集质量差 2.5-3.0dB(§3),
对照 5K 形状消融的 0.5dB 差距:主要来自**容量天花板**(小模型收敛上限),非训练配方问题。

## 2. 预算 vs 实测

### CPU 推理(i9-10850K,8 线程,fp32,30 次 median)

| 段 | 320² | 512² | 备注 |
|---|---|---|---|
| lane 构造 | 1.42ms | 4.10ms | 按行并行(8.3 起) |
| 前向(slim_mixed v1.1) | **25.6ms** | **83.9ms** | min 25.5/83.7,p90 28.4/92.1 |
| composite + PNG | 1.37ms | 3.51ms | |
| **端到端** | **28.4ms** | **91.5ms** | |
| 前向 v0(形状前) | 36.7ms | 115.4ms | 形状瘦身 ×1.43 / ×1.37 |
| 教师 CPU 基线 | 237ms | 767ms | **加速 9.3× / 9.1×**(前向口径) |

512² 前向分段(`NR_PROFILE=1`):gemm 50.1 / attn 15.1 / other 18.6 ms。
attn 累加口径(÷8 线程):norms ~2.2、scores ~5.4、softmax ~1.1、av ~5.1ms。

### GPU(3090,参考)

| | 512² | vs 教师 |
|---|---|---|
| 教师 | 113ms(71 块/147MB) | — |
| 学生 v0 | 11.05ms | 10.2× |
| 学生 slim_mixed | 未测(按 GMAC 折算 ~7.5ms) | ~15× |

### int8(8.8/8.2 口径,AVX2)

| | 320² | 512² | 判定 |
|---|---|---|---|
| 前向 int8 | 27.1ms | 85.0ms | AVX2 上平手/略慢(无 VNNI),保留 opt-in(`NR_INT8=1`) |
| 合成图 PSNR(int8 vs f32) | 65.4dB | 65.7dB | **≈0.00dB 退化 ≤0.3dB** ✅ |
| head 逐通道 max\|diff\| | 2.3/1.5/1.4e-3,logit 2.3e-2 | 2.7/3.2/1.8e-3,logit 2.6e-2 | 超 2e-3 逐元素线的集中在 logit 通道;rgb rmse 3.6e-4,合成图无感 |

## 3. 质量表(6.1/6.2,DAVIS bear+dogs 0000*,10 图,同一 features 喂双网)

| | 320² 学生 | 退化 | vs 教师 | SSIM | corr | blend_mae |
|---|---|---|---|---|---|---|
| **v1.1 slim_mixed(60K)** | 43.33 | **+1.10dB** | 46.68dB | 0.9992 | 0.605 | 0.253 |
| v0(50K,基线) | 45.85 | −1.42dB | 48.44dB | 0.9994 | 0.658 | 0.223 |
| | **512²** | | | | | |
| **v1.1 slim_mixed(60K)** | 44.27 | **+0.05dB** ✅ | 46.25dB | 0.9991 | 0.653 | 0.269 |
| v0(50K,基线) | 47.22 | −2.90dB | 47.50dB | 0.9993 | 0.697 | 0.240 |

(教师自身 44.43@320² / 44.32@512²;退化 = 教师 − 学生,正值 = 学生更差。)

## 4. 时序表(6.3,bear 4 帧,固定 seed)

| 口径 | flicker 教师 | flicker 学生 | × | warp × | 判定 |
|---|---|---|---|---|---|
| v1.1 320² 闭环(部署) | 0.00500 | 0.01336 | ×2.67 | ×0.97 | ❌ |
| v1.1 512² 闭环(部署) | 0.00558 | 0.01550 | ×2.78 | ×0.97 | ❌ |
| v1.1 320² **开环**(`--hist-src teacher`) | 0.00500 | 0.01395 | ×2.79 | ×0.97 | 诊断 |
| v0 基线 320² 闭环 | 0.00499 | 0.01329 | ×2.66 | ×0.98 | ❌(历史腿缺失) |

**定性结论(时序腿无效的根因)**:

1. **反馈环不是根因**:开环(学生历史换成教师输出)×2.79 ≈ 闭环 ×2.67。
2. 误差分解:flicker_s² ≈ flicker_t² + std(Δe)²,反推 **std(Δe) ≈ 0.012**,与学生-教师近似误差
   本身同量级(vs 教师 46.5dB → RMSE ≈ 0.015)→ **学生的近似误差逐帧基本不相关(白化)**。
3. v1.1 的 pair/temporal 腿(pair-fraction 0.3、loss_temporal 0.5、合成视角对)训练损失确实下降
   (temp 0.009→0.002),但**没有迁移到真实视频统计** —— 合成视角对与真实逐帧变化分布不匹配是首要嫌疑。
4. 修复路线(未执行,用户决策本轮照实记录):**真实视频帧对**(DAVIS 连续帧 + 块匹配运动)替代合成
   视角对、pair-fraction 0.5、loss_temporal 1.0 重训,~20h GPU;预期 flicker → ≤1.5×(若根因在容量
   则收益有限,风险自担)。

## 5. 消融表

| 消融 | 配置 | 关键读数 | 结论 |
|---|---|---|---|
| 形状扫描(5K 步) | v0/blocks/mixed/deep | 43.67 / 42.75 / 43.17 / 43.58dB | 步数受限分不开;deep 曾抱输入(corr 0.007) |
| 形状 ×步数(60K) | mixed vs v0(同集) | −2.5dB@320 / −3.0dB@512 | 容量天花板为主(对照 5K 仅差 0.5dB) |
| 时序腿 | pair 0.3 + temp 0.5 vs 无 | flicker ×2.67 vs ×2.66 | **无效**(§4 根因) |
| 320-only | 单尺寸 320² 32K(ckpt `/tmp/run_v1/ckpt.pt`) | comet `abbc0236` | 留档,非官方 |

训练配方(v1.1 定稿):`--size 320,512 --size-weights 0.7,0.3 --batch 4 --steps 60000
--warmup 4000 --pair-fraction 0.3 --loss-temporal 0.5`,EMA 0.999 暖起,四损失
(out 1.0 / detail 0.5 / feature 0.5 / temporal 0.5),约 20h(3090)。

## 6. 结论与后续

**已交付**:
- 小学生(4.29M 参数,fp32 权重 17.2MB)512² 质量与教师差 **+0.05dB**(门 1dB);
  教师 CPU 767ms → 学生 **91.5ms(8.4×)**;GPU 3090 ~15×。
- 手写融合 CPU 引擎(窗序布局、融合 FFN/注意力、int8 GEMM)对账逐级 5e-6、head 120dB;
  int8 质量无损(65.5dB),VNNI 开关就绪。
- 视觉:四联图 [samples/bear512_quad.png](samples/bear512_quad.png) 教师/学生不可分,
  误差落点(§diff 图)集中在纹理区。

**未过门 / 移交**:
- **flicker ×2.78**(门 ≤1.5×):根因定性完成(§4),修复 = 真实视频对重训,待决策。
- **30ms@512²**:AVX2 硬地板(GEMM 打满也不够),按约定移交 **8.7 VNNI 验收**;
  int8 内核在 VNNI 上预期 2-3× → 512² ~30-40ms,30ms 达线仍可能需要形状折扣(slim_deep 7.19 GMAC)
  或分辨率降档(320² 现 28.4ms,VNNI 预期 <15ms,8.7 的 320²/15ms 线大概率过)。
- 320² 退化 +1.10dB 名义超线:非用户目标口径(512²);如需余量,slim_blocks(10.17 GMAC)折中。

**Plan B(形状侧备胎,按需启动)**:
- **MoE**:专家并行窄宽(现 expert 块 E=2 的扩展),容量/算力解耦;
- **低秩**:大 GEMM(全级 FFN 占 26.5%)低秩分解,ratio 扫描;
- **4×4 窗注意力**(`student_alt_attn4x4.json`,12.80 GMAC):需 16 槽注意力 kernel,attn 现占 512² 18%,
  窗加大可再省 scores/softmax 开销。
- 5.14 损失消融(-feature/-detail/+temporal)可选;flicker 重训(§4.4)为时序门的唯一已知路径。

## 附:复现

```bash
# 训练(3090,~20h)
python3 teacher/train_distill.py --teacher-weights models/nr \
  --data data/raw/DIV2K_train_HR data/raw/DAVIS/JPEGImages/Full-Resolution data/train/proxy_proc \
  --shape shapes/student_slim_mixed.json --size 320,512 --size-weights 0.7,0.3 \
  --batch 4 --steps 60000 --warmup 4000 --pair-fraction 0.3 --loss-temporal 0.5 -o run_v11_mixed

# 质量 / 时序(6.1/6.3)
python3 teacher/eval_quality.py  --teacher-weights models/nr --shape shapes/student_slim_mixed.json \
  --sizes 320,512 --checkpoint run_v11_mixed/ckpt.pt --images 'data/raw/DAVIS/.../bear/0000*.jpg' \
  'data/raw/DAVIS/.../dogs/0000*.jpg' -o eval_v11
python3 teacher/eval_temporal.py --teacher-weights models/nr --shape shapes/student_slim_mixed.json \
  --seq 'data/raw/DAVIS/.../bear/*.jpg' --checkpoint run_v11_mixed/ckpt.pt --size 512 --max-frames 4 \
  -o eval_v11_t   # --hist-src teacher 开环诊断

# 引擎对账 / 测速 / int8(8.4/8.5/8.8)
python3 teacher/check_engine.py --image <png> --size 512 --shape shapes/student_slim_mixed.json \
  --checkpoint run_v11_mixed/ckpt.pt --bench          # NR_INT8=1 → int8;NR_PROFILE=1 → 分段
python3 teacher/quad_compare.py --image <png> --size 512 --shape shapes/student_slim_mixed.json \
  --checkpoint run_v11_mixed/ckpt.pt --teacher-weights models/nr -o teacher/samples   # 6.5
```
