# 服务器运行状态(2026-10-18)

本文描述训练服务器上**实际**的代码、数据、配置和产物状态,供接手或复查时对照。
所有数字均从服务器现场采集,不是回忆值。

## 1. 机器与环境

| 项 | 值 |
|---|---|
| 主机 | `yvonchen-LC0`(IP `192.168.3.32`,用户 `koth`) |
| 系统 | Ubuntu 20.04,内核 `5.15.0-139-generic` |
| GPU | NVIDIA GeForce RTX 3090,24576 MiB |
| 驱动 | 535.171.04(未升级) |
| 训练进程 | PID `251423`,常驻约 4.6 GB 内存,GPU 21043 MiB / 95% / 82 °C |

### Python 环境(两套并存)

| 路径 | 版本 | 用途 |
|---|---|---|
| `/f/yanr/keep_tmp/perf_20261017/torch2141/bin/python3` | Python 3.14.7 + Torch 2.14.1+cu126(CUDA 12.6,cuDNN 91002) | **当前训练用** |
| `python3`(系统) | Python 3.8.10 + Torch 2.4.1+cu121 | 旧环境,保留 |

Torch 2.14.1 环境是 2026-10-17 为提速单独装的,装了 `comet_ml`。

### 磁盘

| 挂载点 | 容量 | 已用 | 可用 |
|---|---:|---:|---:|
| `/f`(NVMe) | 932G | 705G | 228G |
| `/`(根分区) | 916G | 853G | **17G(99%)** |

> 根分区只剩 17G。当前 run 的产物全部写在 `/f` 上,不受影响;但往 `/home/koth`
> 再存 checkpoint(每个约 1.2G)会很快吃满。清理时优先看 `/home/koth/anaconda3`
> (86G)与 `/home/koth/文档`(38G)。

## 2. 代码

### 位置与版本

- 代码目录:`/home/koth/klss`(**不是 git 仓库**,是从本地 rsync 过去的副本)
- 唯一可追溯的版本来源是本地仓库 `/Users/kothchen/code/klss`,当前 HEAD `e65d847`→`2df7d60`
- **服务器代码已与本地 HEAD 分叉**(2026-10-17 22:29 – 10-18 00:48 期间被改动)。
  不能再用"哈希一致"当作假设,每次续训前必须重新核对。

| 文件 | 本地 HEAD | 服务器 | 差异性质 |
|---|---|---|---|
| `teacher/train_distill.py` | `b1eaa5a4…` | `46487c71…` | 服务器保留了教师**分块并行**标注路径(`teacher_workers>1`),本地在 36e7a5b 简化为串行;另有 4 个 CLI 参数 |
| `teacher/nr_student.py` | `d5b76d62…` | `2bc8d510…` | `import torch._dynamo` + 一个 `@torch._dynamo.disable` |
| `teacher/nr_shared_moe.py` | `69457c6d…` | `f0676238…` | `row_window.clamp(0, batch*nw*nh-1)`,注明为 Inductor 间接索引检查 |
| `teacher/degrade.py` | `8f3c159e…` | `71d83311…` | 仅 `main()`:新增 `--ordinal-offset`、`out_dir` 分片逻辑。训练调用的函数未变 |
| `teacher/nr_cuda.py` | `8f0b3c60…` | `f47a9c54…` | 编译期 `NR_CUDA_ARCH` 环境变量,运行期不使用 |
| `teacher/moe_training.py` | `1817fda9…` | `1817fda9…` | **一致** |
| `teacher/probe_input_regime.py` | 无 | 有 | 服务器多出的诊断脚本 |

> 服务器版**包含 v4.5c 的全部改动**(align 修复、`restore_optimizer`、
> `moe_bias_span_cap`、新损失默认值、`flush=True`、`out_logit` 列),
> 与本地是同一份改动的两个分支。因此**不要用本地文件覆盖服务器**,
> 否则会丢掉并行教师路径。

### 分叉的数值影响(已实测)

用同一 checkpoint(44k)、同一张图,在"服务器版"和"仅把 `nr_shared_moe.py`
换成本地版"两个目录下跑同一段前向:

```
md5      : a7ab348dc000345b2560ae4baeddf0b1   (两边完全相同)
sum/abs  : -683174.8750000000 / 706981.7500000000
```

**输出逐比特一致**,`row_window` 的 clamp 只影响被 mask 的尾 lane。
另外那两处是 `torch.compile` 专用(`student_compile=false` 时不起作用)。

### 续训时的口径选择

`v45d` 续训把 `teacher_workers` 设为 **1**,让服务器版的并行路径退化为串行
(`annotate_pool = None`),与已跑完 24k 步的代码路径保持一致。要提速可以
改成 2 试,但该路径在 3090 上未验证。

### 目录内容

| 路径 | 内容 |
|---|---|
| `/home/koth/klss/teacher/` | 65 个 `.py`,67M(含训练、bench、check、eval 脚本) |
| `/home/koth/klss/teacher/models/nr` | 教师权重 |
| `/home/koth/klss/shapes/` | 9 个学生结构 JSON,当前用 `student_shared_moe_v1.json` |
| `/home/koth/klss/run_v45d_cfg.json` | **当前 run 的配置**(texstat 2.5 续训) |
| `/home/koth/klss/run_v45c_cfg.json` | v4.5c 配置(texstat 1.0),保留对照 |
| `/home/koth/klss/run_v45_cfg.json` | 上一版配置(v4.5b),保留对照 |

> 注意:服务器不是 git 仓库,**改动代码必须从本地 rsync/scp 推送**,不能指望在
> 服务器上 `git pull`。

## 3. 数据

入口是 8 个数据根目录,**顺序固定**(与 `teacher/benchmarks/v45_training_3090_20261017.json`
的 `data/roots` 一致),`FrameSource` 递归扫描:

| # | 根目录 | 图片数 | 备注 |
|---|---|---:|---|
| 1 | `/f/yanr/rs/DIV2K_train_HR` | 800 | DIV2K |
| 2 | `/f/yanr/rs/Flickr2K` | 2650 | Flickr2K |
| 3 | `/f/yanr/rs/Full-Resolution` | 6208 | DIV2K 全分辨率版本 |
| 4 | `/f/yanr/rs/proxy_proc` | 1600 | 预退化代理图 |
| 5 | `/f/yanr/rs/rgb` | 2674 | 游戏 RGB |
| 6 | `/f/yanr/rs/gameplay` | 10803 | 游戏截图 |
| 7 | `/f/yanr/rs/portraits` | 4000 | 人像 |
| 8 | `/f/yanr/rs/frames_2k` | **1847(软链接)** | 指向 `/f/yanr/rs/frames` |
| | **合计** | **30582** | 与日志 `train: 30582 files` 吻合 |

> `frames_2k` 里全是符号链接(`ln -s /f/yanr/rs/frames/*.jpg`),所以
> `find -type f` 数出来是 0。别被这个数字误导——`FrameSource` 能正常读取,
> 实际计入 1847 张。

未使用的相邻目录: `/f/yanr/rs/gameir`(空)、`/f/yanr/rs/frames`(11078,被
`frames_2k` 引用但未单独作为根)。

## 4. 训练配置(当前 run_v45d)

### 结构

| 键 | 值 |
|---|---|
| `shape` | `shapes/student_shared_moe_v1.json`(shared-routed-moe-v1) |
| `student_precision` | `fp32` |
| `router_jitter` | `0.0`(确定性) |
| `student_compile` | `false` |

### 采样

| 键 | 值 |
|---|---|
| `size` / `size_weights` | `[320, 512]` / `[0.7, 0.3]` |
| `batch` | 8 |
| `batch_rows` / `teacher_batch_rows` | 1179648 / 589824 |
| `pair_fraction` | 0.35 |
| `crop_prob` / `degrade_prob` | 0.5 / 0.5 |
| `style_range` / `tone_range` / `structure_range` | `[2,2]` / `[1,1]` / `[1,1]` |
| `skin_range` / `automask_prob` | `[-1,-1]` / 1.0 |
| `seed` | 20261007 |

### 优化(v45d 续训段)

| 键 | 值 | 说明 |
|---|---|---|
| `steps` | **56000** | 原 60000 提前收;44k 起再跑 12k |
| `lr` / `warmup` | **5e-5** / 4000 | 原 2e-4;从 1.2e-5 抬上来才推得动幅度 |
| `lr_hold` / `lr_decay` / `lr_floor` | **0.85 / 0.15** / 1e-5 | 44001–47600 平 5e-5,再余弦降到 56000 触底 |
| `clip` / `wd` | 1.0 / 0.01 | — |
| `ema` / `ema_every` | 0.999 / 4 | — |
| `teacher_workers` | **1** | 服务器版并行路径退化为串行,与已跑的 24k 步保持同路径 |

### 损失(v45d)

| 键 | 值 | 相对 v4.5c |
|---|---:|---|
| `loss_out` | 1.0 | — |
| `loss_detail` | 0.5 | — |
| `loss_feature` | 0.5 | — |
| `loss_temporal` | 1.0 | — |
| `loss_logit` | 0.3 | — |
| `loss_texture` | 3.0 | — |
| `loss_texstat` | **2.5** | 1.0 → 2.5(唯一的目标函数改动) |
| `texstat_boost` | 0.0 | — |
| `texture_floor` / `texture_knee` | 0.4 / 0.005 | — |
| `action_floor` / `action_knee` | 0.25 / 0.006 | — |

### MoE

| 键 | 值 |
|---|---|
| `moe_aux` / `moe_z` | 0.0 / 0.0 |
| `moe_bias_rate` | 1e-3 |
| `moe_bias_span_cap` | **2.0**(v4.5c 新增) |

### 日志

| 键 | 值 |
|---|---|
| `log_every` / `vis_every` / `ckpt_every` | 100 / 2000 / 2000 |

## 5. 产物

### 当前运行目录 `/f/yanr/keep_tmp/run_v45d`

| 文件 | 说明 |
|---|---|
| `ckpt.pt` | 1.2G,最近一次 checkpoint(每 2000 步覆盖) |
| `config.json` | 本次 run 的实际配置(启动时落盘,含运行时重算的 `source`) |
| `log.csv` | 每 100 步一行;列 = `step,lr,loss,out,detail,feature,temporal,psnr_teacher,ms,tex,stat,out_logit` |
| `routing.jsonl` | 每 100 步路由遥测(逐层 bias/负载/梯度范数) |
| `train.log` | 标准输出;v4.5c 起 `print(flush=True)`,实时可读 |
| `vis-046000/` … | 每 2000 步一组(proxy/teacher/student/diff_x12);**样本随机,只用于快速看图,不做跨点对比** |

### 上一阶段 `/f/yanr/keep_tmp/run_v45c`(v4.5c,停在 44000)

Comet: https://www.comet.com/koth-chen/dlss-student/8d4f841dba83488fb7ccad1dc3d15b9b

### 归档

| 路径 | 大小 | 内容 |
|---|---:|---|
| `/home/koth/keep_tmp/v45b_20k/` | 1.2G | **续训起点**:v4.5b 的 20k ckpt + config + log |
| `/f/yanr/keep_tmp/ckpt_v45b_20k.pt` | 1.2G | 同上(副本) |
| `/f/yanr/keep_tmp/ckpt_v45b_10k.pt` | 1.2G | v4.5b 10k 快照 |
| `/f/yanr/keep_tmp/run_v45b/` | 1.3G | v4.5b 完整运行目录(停在 21300 步) |
| `/home/koth/keep_tmp/v45c_44k/` | 1.2G | **v45d 的续训起点**:v4.5c 的 44k ckpt + config + log(md5 `475067aa…`) |
| `/f/yanr/keep_tmp/perf_20261017/` | 6.5G | 提速实验的独立 Torch 环境与产物 |
| `/home/koth/keep_tmp/run_v43_ckpt44k.pt` 等 | 各 969M | v4.3/v4.44a/v4.44b 历史 ckpt |

## 6. 现场校验

```bash
# 进程与进度(PID 每次启动都会变,先 pgrep)
pgrep -af train_distill
tail -3 /f/yanr/keep_tmp/run_v45d/log.csv

# GPU
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv

# 代码现状(与本地 HEAD 已分叉,只做记录,不要假设一致)
sha256sum /home/koth/klss/teacher/train_distill.py | cut -c1-16

# 路由健康(最近一步)
tail -1 /f/yanr/keep_tmp/run_v45d/routing.jsonl | python3 -m json.tool | head -40
```

### 固定图集纹理评估(跨 checkpoint 唯一可信口径)

```bash
cd /home/koth/klss/teacher && python3 eval_texture.py \
  --images /f/yanr/rs/gameplay/Cyberpunk_2077___dvljiQgpCAE_frames000000028.png \
           /f/yanr/rs/gameplay/Cyberpunk_2077___dvljiQgpCAE_frames000005507.png \
           /f/yanr/rs/gameplay/Ghost_of_Tsushima___UOheOYV-H2U_frames000008006.png \
           /f/yanr/rs/portraits/female/female_00000.jpg /tmp/cpwoman_smooth.png \
  --teacher-weights /home/koth/klss/models/nr --width 320 --height 512 \
  --ckpt base20k=/home/koth/keep_tmp/v45b_20k/ckpt.pt \
  --ckpt v45d=/f/yanr/keep_tmp/run_v45d/ckpt.pt -o /tmp/eval_now
```

单次约 1 分钟、显存约 1GB,可在训练进行时并行跑(实测不干扰)。

### 常用分析脚本(在服务器 `/tmp/`)

这些是排障时写的临时脚本,不在版本库里:

| 脚本 | 用途 |
|---|---|
| `vis_bands.py` | 纹理复刻口径(fine/coarse 双频带、扣量化噪声);样本随机,仅单点用 |
| `vis_signed.py` | 带符号的"加/减纹理"方向诊断 |
| `cmp_forward.py` | 两个代码目录下前向输出的逐比特比对 |
| `vis_modulation2.py` | 纹理区 `Ds = a·Dt` 回归、corr |
| `vis_where.py` | `corr(改动, 输入高频)`,看"哪儿改" |
| `routing_health.py` | 路由 bias span / 负载 / 熵 |
| `comet_features.py` | 拉 comet 的逐点 `feat/*` 序列 |
| `logwin.py` | 两版日志同窗口中位数对比 |
| `loss_breakdown.py` | 损失分项与权重的比值 |

## 7. 当前进度(采集于 2026-10-17 21:46)

| 项 | 值 |
|---|---|
| 当前 run | `v45d`,从 44k 续训,目标 56000 |
| 步数 | 44300 / 56000 |
| 速度 | ~1.8 s/步,显存 23515 MiB(稳定,不增长) |
| lr | 5e-5(44001–47600),之后余弦降到 56000 的 1e-5 |
| `feat`(窗口中位数) | 0.858(20k)→ **0.135**(21k-21.9k)→ 0.009(42k-44k) |
| 纹理 corr / 幅度比 | 见下(固定图集口径) |

### 纹理口径的两次纠正

1. **vis 采样是随机的**:训练每 2000 步抓 `chunk[-1]['proxy']`,每步不同图
   (实测 22k/30k/42k/20k 的 `proxy.png` 哈希全不同)。所以跨 checkpoint 比
   corr/amp 是在比不同图片,样本内容差异盖过模型进步。**不要再用 vis 目录做
   跨点对比**,改用 `teacher/eval_texture.py`(固定图集)。
2. **修正后的固定图集结论**(5 张图,style2/tone1/struct1/skin-1/automask1):

| ckpt | fine corr | coarse corr | fine amp | coarse amp | s-vs-t |
|---|---:|---:|---:|---:|---:|
| 10k | 0.58 | 0.57 | 0.65 | 0.61 | 30.99 |
| 20k | 0.60 | 0.59 | 0.80 | 0.71 | 31.17 |
| 42k | 0.68 | 0.69 | 0.51 | 0.56 | 31.12 |

即:v4.5c 把**形状相关性**提上去了,但把**幅度**砍掉约 30%(texstat 3.0→1.0
的直接后果)。44k 的固定图集实测(2 张图)coarse corr 0.74 / amp 0.43,方向仍对
(`mean_ds = -0.02`,教师 `+0.10`),幅度不足依旧。

`v45d` 就是针对这一点:`loss_texstat` 1.0 → **2.5**(无 boost),其余不动,
用 5e-5 的 lr 跑 12k 步把它补回来。验收判据:幅度比回到 `[0.8, 1.2]`
且 corr 不跌回 0.6 以下。
