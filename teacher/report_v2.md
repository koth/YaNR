# 学生模型 v2.0(student_moe_v0)60K 验收报告

模型:`student-student_moe_v0-step59999`,包 `weights/student_moe_v0`
(zip sha256 `3d3fc82b9c92126686d6582f22f11d83dff36608cdbe5a1b17e5d188f749a404`)。
教师:OpenDLSS-NR 复原(`models/nr`)。口径工具:`eval_conditions.py` / `eval_quality.py` /
`eval_temporal.py` / `check_moe.py` / `check_package.py`(本仓库 `teacher/`)。

## 1. 模型与训练

- **v0 宽度 + 全级 MoE**:channels 32-256、ViT 512/1024×3 块、每级 kind=moe **E=8 top-1 配额平衡**
  (名额 ⌈W/E⌉、弱 margin 让位、归一化门控 gate=E·p、router 前置 LN、z-loss 1e-2、aux 0.02)。
  **62,831,273 参数**(fp32 251.3MB),逐像素计算量 = v0 dense:**13.71 GMAC @512²**(教师 ~69 GMAC)。
- 训练:60K 步、batch 4、EMA 0.999、lr 2e-4(cosine、warmup 4K);真相邻帧对(pair 0.5,
  teacher-forcing 历史腿 + loss_temporal 1.0)+ 在线退化(crop/degrade 0.5)+ **条件全量程采样**
  (style [0,127]、tone/structure [0,1]、automask 0.3)。
- 数据 7 源 31,074 张(DIV2K / DAVIS Full-Resolution / proxy_proc / Flickr2K / GameIR-SR /
  真实截图 / UHD),**游戏帧 64%**;批内 4 线程并行准备(确定性逐位不变,种子 (step,bi))。
- 训练盘面:loss 22.8→3.9、feat 20.6→3.5、psnr(teacher) 24→32;路由 util_max 0.17、
  util_worst 0.37、dead 3%、z 0.09,check_moe 9 钉 PASS。

## 2. 条件控件全量程验收(eval_conditions,512²,5 图 = bear/proxy/合成 + 2 游戏帧;s-vs-t = 学生 vs 教师合成图)

| style | 0 | 2 | 4 | 8 | 16 | 32 | 64 | 96 | 127 | all-max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **v2.0** | 39.4 | 41.1 | 39.2 | 41.3 | 40.1 | 37.0 | 34.1 | 35.9 | 29.8 | 26.9 |
| v1.1(60K) | 43* | 43* | — | — | 23.9 | 18.9 | 9.9 | 8.9 | — | 9.1 |

\* v1.1 只在近零档(style lane ≤0.023)训练,IN 档偏科;style≥16 全面塌方(旋钮不可用)。
v2.0 全量程 34-43 无塌方;tone/struct/skin+automask 全档 36-43;游戏帧与照片读数持平
(部分档反超);全表 corr 0.4-0.88(all-max 0.88)。

## 3. Known issues(读数备注)

| 项 | 读数(参考线) | 状态 |
| --- | --- | --- |
| 6.2 退化 vs 输入(512²,默认工作点 1.5/0.5/0.35,bear/camel 8 图) | +5.06dB(参考 ≤1dB) | 偏高,记入 known issues |
| 6.3 flicker(512²,DAVIS+GameIR 序列,新基线) | ×1.74(参考 ≤1.5×;warp ×1.02) | 略高,轻度可见,记入 known issues |
| 7.3 打包往返 | 229 张量逐位一致、前向 torch.equal | 通过 |
| check_moe | 9 钉全绿 | 通过 |

**根因(同一个)**:学生模仿误差 E(s-vs-t ~39-40dB @默认工作点)> 教师自身动作幅度(教师 vs 输入
~44dB)。默认工作点上教师动作小,E 相对显形 → 6.2(改动偏大)与 6.3(E 时间分量 → 轻微 shimmer)同发。
全量程采样摊薄了自然档的拟合精度(**采样问题非容量问题**,v1.1 近零档专训可达 44.4dB)。

对照口径:v1.1 在**同集同点**只有 +1.4dB(bear −0.26 / camel +3.07),v1.1 历史的 +0.05dB 是更软
的旧 10 图集,不可比;flicker 旧读数 ×2.7/×2.78 受运动场单位/方向 bug 污染,本报告 ×1.74 是修复
后的有效基线。flicker 绝对值:学生 ~2.2-2.5/255、教师 ~1.4/255(多出 ~1.8/255 RMS,轻度可见,
运动掩蔽强;ghosting 干净)。

**处置**:按用户决策照发,两项记入 known-issues。收口路径(备选,~6-9h):从 60K ckpt 低 LR 微调
10-15K 步、采样收窄近零档(style [0,3]、tone [0.3,0.7]、structure [0.2,0.8]),预期默认工作点
s-vs-t 39→43+、flicker →~1.4×;效果档回退用全表对照量化,>2dB 再升级混采重训(30h)。

## 4. 性能与引擎

- torch 参考(eager,512²):服务器侧中位 62.8ms/帧(run_nr,5 帧)。
- CPU 引擎(AVX2/OpenMP)MoE 支持(grouped GEMM)与 **8.7 VNNI 验收**未进本包;shape.json 注记
  预估:512² AVX2 ~125ms、VNNI+int8 45-65ms(验收以实测为准)。int8 阶梯沿用 v1.1(NR_INT8=1)。

## 5. 复现

```sh
# 训练配方 = ckpt 内 config(shape/data 见 §1;seed 20261007,确定性 (step,bi))
python3 train_distill.py --shape ../shapes/student_moe_v0.json --steps 60000 --batch 4 \
  --pair-fraction 0.5 --loss-temporal 1.0 --moe-aux 0.02 --moe-z 0.01 -o runs/run_final

# 验收(本文 §2/§3 口径)
python3 eval_conditions.py --images <5图> --shape ../shapes/student_moe_v0.json \
  --checkpoint runs/run_final/ckpt.pt --base-style 1.5 -o eval_final_60k
python3 eval_quality.py  --shape ../shapes/student_moe_v0.json --sizes 512 \
  --checkpoint runs/run_final/ckpt.pt --images 'bear/0000*.jpg' 'camel/0000*.jpg' -o eval_quality_60k
python3 eval_temporal.py --shape ../shapes/student_moe_v0.json --checkpoint runs/run_final/ckpt.pt \
  --size 512 --max-frames 8 --seq 'bear/0000*.jpg' 'camel/0000*.jpg' '<gameir>/00/1440p/*.rgb.jpg' -o eval_temporal_60k

# 打包(7.1-7.3)
python3 convert_weights.py --shape student runs/run_final/ckpt.pt -o weights/student_moe_v0
python3 check_package.py --package weights/student_moe_v0 --checkpoint runs/run_final/ckpt.pt
```
