# 学生模型 v3.0(配方重训)60K 验收报告

模型:`student-student_moe_v0-step59999`(与 v2.0 同架构:62,831,273 参数、13.71 GMAC @512²、
全级 MoE E=8),包 `weights/student_v3`(zip sha256
`dd85e1b95dc1e251b2081340ccf6a887e94d495bc09c629be0d5794b62202e4e`)。
验收工具同 v2.0(`eval_conditions/eval_quality/eval_temporal/check_moe/check_package`)。
验收集(2026-10-13 起固定 4 图):bear / proxy_proc / GameIR 帧 ×2(旧合成探针图随 /tmp 清理丢失)。

## 1. 本版配方(用户口径:低水平修改不需学,学就学改动较大的)

- **教师改动加权损失**:逐像素 `w = 0.1 + 0.9·Δ²/(Δ²+0.02²)`,Δ = 教师 |合成−输入| ——
  强改动区满权、弱改动区压地板,不做均值归一化(梯度占比 = 改动幅度占比;冒烟强/弱 ~10×)。
  blend-logit 幅度阀升为独立项(0.5 内联 → 默认 1.0)。
- **LR 分段封底**:warmup 4K → 2e-4 平段(至 30K)→ 余弦降到 1e-5(45K)→ 封底不归零。
- **吞吐改造(2.4×)**:真 batch(行主序叠放,batch=1 与逐样本逐位一致)+ 教师标注流水
  (预取线程 ×3、线程本地 CUDA 流、`annotate` 全设备同步改流本地)。有效步速
  4.3s → **~1.8s/step(batch 8)**,60K ≈ 30h;GPU 39% → 99%。
- 数据/其余超参同 v2.0(7 源 31K、64% 游戏帧、pair 0.5、时序损失 1.0、全量程条件采样)。

## 2. 条件全量程验收(eval_conditions,512²,4 图均值 s-vs-t)

| style | 0 | 2 | 4 | 8 | 16 | 32 | 64 | 96 | 127 | all-max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **v3.0(60K)** | 37.8 | 40.9 | 40.0 | 39.8 | 39.5 | **37.4** | 34.6 | **35.9** | **30.6** | **27.9** |
| v2.0(60K) | 38.5 | 40.9 | 40.0 | 40.2 | 39.8 | 37.0 | 34.6 | 34.9 | 29.8 | 26.6 |

tone/struct/skin+am 全档 36.3-42.4;**24 行中 12 行超过 v2.0 终态**,其余差距 ≤0.76dB。
消长形态即配方意图:强改动档(style 32/96/127、all-max)+0.4~1.3dB,弱改动档(skin、
tone 0.75-1.0)让 ~0.7dB。all-max corr **0.919**(v2.0 0.883)。

## 3. 读数备注(known issues)

| 项 | v3.0 | v2.0 | 参考线 | 状态 |
| --- | --- | --- | --- | --- |
| 6.2 退化 vs 输入(512²,默认工作点) | +5.61dB | +5.06dB | ≤1dB | 偏高,记入 known issues(用户决策:弱档不追) |
| 6.3 flicker(512²,DAVIS+GameIR) | **×1.44**(warp ×1.02) | ×1.74 | ≤1.5× | **达标** |
| blend_mae | 0.546 | 0.617 | — | 改善 |
| 打包往返 | 229 张量逐位一致 | ✓ | 逐位 | 通过 |
| check_moe(10 钉,含批量钉) | 全绿 | ✓ | — | 通过 |

flicker 从 ×1.74 → ×1.44 直接来自改动加权(模仿误差是闪烁源,误差降则闪减)。
6.2 的默认工作点问题是弱改动档的拟合精度(教师动作 ~0.6/255 量级),按用户口径不追,
收口路径同 report_v2 §3(近零档微调,~6-9h)。

## 4. 复现

```sh
# 配方 = ckpt 内 config(action_floor/knee、lr_hold/decay/floor、loss_logit 默认值)
python3 train_distill.py --shape ../shapes/student_moe_v0.json --steps 60000 --batch 8 \
  --warmup 4000 --pair-fraction 0.5 --loss-temporal 1.0 --teacher-workers 3 \
  --data <7 源> -o runs/run_v3

# 验收(§2/§3 口径)+ 打包
python3 eval_conditions.py --images <4 图> --shape ../shapes/student_moe_v0.json \
  --checkpoint runs/run_v3/ckpt.pt --base-style 1.5 -o eval_v3_60k
python3 eval_quality.py  --shape ../shapes/student_moe_v0.json --sizes 512 --limit 8 \
  --checkpoint runs/run_v3/ckpt.pt --images 'bear/0000*.jpg' 'camel/0000*.jpg' -o eval_quality_v3
python3 eval_temporal.py --shape ../shapes/student_moe_v0.json --checkpoint runs/run_v3/ckpt.pt \
  --size 512 --max-frames 8 --seq 'bear/0000*.jpg' 'camel/0000*.jpg' '<gameir>/00/1440p/*.rgb.jpg' \
  -o eval_temporal_v3
python3 convert_weights.py --shape student runs/run_v3/ckpt.pt -o weights/student_v3
python3 check_package.py --package weights/student_v3 --checkpoint runs/run_v3/ckpt.pt
```
