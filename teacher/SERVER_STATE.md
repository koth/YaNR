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
- 唯一的版本来源是本地仓库 `/Users/kothchen/code/klss`,当前 HEAD `0342584`
- 现场核对:服务器上 4 个核心文件的 sha256 与本地 HEAD **完全一致**

| 文件 | sha256(前 16 位) |
|---|---|
| `teacher/train_distill.py` | `b1eaa5a46a3c27f9` |
| `teacher/moe_training.py` | `1817fda930201ef7` |
| `teacher/nr_student.py` | `d5b76d6203b892dc` |
| `teacher/nr_shared_moe.py` | `69457c6d059aabc5` |

### 目录内容

| 路径 | 内容 |
|---|---|
| `/home/koth/klss/teacher/` | 65 个 `.py`,67M(含训练、bench、check、eval 脚本) |
| `/home/koth/klss/teacher/models/nr` | 教师权重 |
| `/home/koth/klss/shapes/` | 9 个学生结构 JSON,当前用 `student_shared_moe_v1.json` |
| `/home/koth/klss/run_v45c_cfg.json` | **当前 run 的配置**(与本地仓库同名文件一致) |
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

## 4. 训练配置(当前 run_v45c)

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

### 优化

| 键 | 值 |
|---|---|
| `steps` | 60000 |
| `lr` / `warmup` | 2e-4 / 4000 |
| `lr_hold` / `lr_decay` / `lr_floor` | 0.5 / 0.25 / 1e-5 |
| `clip` / `wd` | 1.0 / 0.01 |
| `ema` / `ema_every` | 0.999 / 4 |

### 损失(v4.5c 再平衡后)

| 键 | 值 | 相对 v4.5b |
|---|---:|---|
| `loss_out` | 1.0 | — |
| `loss_detail` | 0.5 | — |
| `loss_feature` | 0.5 | — |
| `loss_temporal` | 1.0 | — |
| `loss_logit` | **0.3** | 1.0 → 0.3 |
| `loss_texture` | **3.0** | 1.5 → 3.0 |
| `loss_texstat` | **1.0** | 3.0 → 1.0 |
| `texstat_boost` | **0.0** | 3.0 → 0.0 |
| `texture_floor` / `texture_knee` | 0.4 / 0.005 | floor 0.25 → 0.4 |
| `action_floor` / `action_knee` | **0.25 / 0.006** | 0.1/0.02 → 0.25/0.006 |

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

### 当前运行目录 `/f/yanr/keep_tmp/run_v45c`(1.3G)

| 文件 | 说明 |
|---|---|
| `ckpt.pt` | 1.2G,最近一次 checkpoint(每 2000 步覆盖) |
| `config.json` | 本次 run 的实际配置(启动时落盘) |
| `log.csv` | 每 100 步一行;列 = `step,lr,loss,out,detail,feature,temporal,psnr_teacher,ms,tex,stat,out_logit` |
| `routing.jsonl` | 每 100 步路由遥测(逐层 bias/负载/梯度范数) |
| `train.log` | 标准输出;v4.5c 起 `print(flush=True)`,实时可读 |
| `vis-022000/` | 每 2000 步一组(proxy/teacher/student/diff_x12) |

Comet: https://www.comet.com/koth-chen/dlss-student/8d4f841dba83488fb7ccad1dc3d15b9b

### 归档

| 路径 | 大小 | 内容 |
|---|---:|---|
| `/home/koth/keep_tmp/v45b_20k/` | 1.2G | **续训起点**:v4.5b 的 20k ckpt + config + log |
| `/f/yanr/keep_tmp/ckpt_v45b_20k.pt` | 1.2G | 同上(副本) |
| `/f/yanr/keep_tmp/ckpt_v45b_10k.pt` | 1.2G | v4.5b 10k 快照 |
| `/f/yanr/keep_tmp/run_v45b/` | 1.3G | v4.5b 完整运行目录(停在 21300 步) |
| `/f/yanr/keep_tmp/perf_20261017/` | 6.5G | 提速实验的独立 Torch 环境与产物 |
| `/home/koth/keep_tmp/run_v43_ckpt44k.pt` 等 | 各 969M | v4.3/v4.44a/v4.44b 历史 ckpt |

## 6. 现场校验

```bash
# 进程与进度
ps -o pid,etime -p 251423
tail -3 /f/yanr/keep_tmp/run_v45c/log.csv

# GPU
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv

# 代码一致性(与本地 HEAD 比对)
sha256sum /home/koth/klss/teacher/train_distill.py | cut -c1-16

# 路由健康(最近一步)
tail -1 /f/yanr/keep_tmp/run_v45c/routing.jsonl | python3 -m json.tool | head -40
```

### 常用分析脚本(在服务器 `/tmp/`)

这些是排障时写的临时脚本,不在版本库里:

| 脚本 | 用途 |
|---|---|
| `vis_bands.py` | 纹理复刻口径(fine/coarse 双频带、扣量化噪声) |
| `vis_modulation2.py` | 纹理区 `Ds = a·Dt` 回归、corr |
| `vis_where.py` | `corr(改动, 输入高频)`,看"哪儿改" |
| `routing_health.py` | 路由 bias span / 负载 / 熵 |
| `comet_features.py` | 拉 comet 的逐点 `feat/*` 序列 |
| `logwin.py` | 两版日志同窗口中位数对比 |
| `loss_breakdown.py` | 损失分项与权重的比值 |

## 7. 当前进度(采集于 2026-10-17 21:46)

| 项 | 值 |
|---|---|
| 步数 | 22100 / 60000 |
| 速度 | ~1.7 s/步 |
| 剩余 | ~11.4 小时,预计次日上午收工 |
| `feat`(窗口中位数) | 0.858(20k)→ **0.135**(21k-21.9k) |
| `out` | 0.290 → **0.161** |
| 纹理 corr / 幅度比 | 0.44 / 0.91(20k)→ **0.47 / 0.68**(22k) |

最后一行需要关注的:v4.5c 压低了统计项权重,幅度比从 0.91 掉到 0.68。
判据是跑完后幅度比能否回到 `[0.8, 1.2]`;若停在 0.6 以下,说明退化成了
"少造纹理"(v4.1 的失败模式),需要把 `loss_texstat` 从 1.0 回补到 1.5–2.0。
