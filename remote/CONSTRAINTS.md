# KLSS 远端训练约束

更新时间：2026-10-17

## 1. 主机与目录

- 登录目标：`ssh hezm@10.2.0.16`
- 远端主机：`g001`
- 项目根目录：`/data2/hezm/klss`
- 远端不是 git 仓库；代码通过本地 `rsync` 同步，不能假设可以 `git pull`
- `/data2` 可用空间约 9.1T；根分区仅用于系统与用户环境

## 2. GPU 约束（硬约束）

- **只允许使用物理 GPU 6**（从 0 开始计数，即第 7 张卡）
- 所有训练、验证、基准、调试命令都必须带：

```sh
export CUDA_VISIBLE_DEVICES=6
```

- 进程内该卡会显示为逻辑 `cuda:0`，这是正常现象
- 启动前必须检查 `nvidia-smi -i 6`，确认没有其他进程占用
- 禁止使用 `CUDA_VISIBLE_DEVICES=0..5,7` 或任何形如 `0,1,...` 的列表
- 禁止把教师和学生放到不同物理卡；本项目当前按单卡流水线设计

GPU 6 当前型号：NVIDIA A100 80GB PCIe。CUDA 内核必须按 `sm_80` 构建。

## 3. 目录布局

```text
/data2/hezm/klss/
  CONSTRAINTS.md          # 本文件
  code/                   # teacher / engine / OpenDLSS-NR / shapes
  config/                 # 远端训练配置
  data/train/proxy_proc/  # 预退化代理序列
  data/raw/DIV2K_train_HR/
  data/raw/DAVIS/JPEGImages/480p/
  models/nr/              # 教师权重（NVIDIA 条款约束，不对外分发）
  weights/                # 历史学生 checkpoint（按需）
  runs/                   # 训练产物
  logs/                   # nohup 日志、PID、环境信息
  env/                    # uv 创建的 Python 虚拟环境
  uv-cache/               # uv 缓存
```

## 4. Python 与 CUDA 环境

- 使用远端已有 `uv`：`/home/hezm/.local/bin/uv`
- 虚拟环境：`/data2/hezm/klss/env`
- Python：3.10（远端系统 Python）
- PyTorch：`torch==2.14.0+cu126`（驱动 535 不支持 cu130；cu126 已验证可用）
- 原生 CUDA 内核使用 `/usr/local/cuda/bin/nvcc`（CUDA 12.1）
- 编译目标：`NR_CUDA_ARCH=sm_80`
- 禁止使用系统 Python 直接训练；必须激活 `env`
- SSH 复用：`ControlPath=/tmp/klss-ssh`，ControlPersist 2h

## 5. 训练配置

- 配置：`config/run_v46_a100.json`
- 学生结构：`code/shapes/student_shared_moe_v1.json`
- 教师权重：`models/nr`
- 训练步数：50000
- 输入尺寸：320（70%）/ 512（30%）
- batch：10
- teacher workers：1（实测 2 worker 反而略慢）
- 教师 batch 行预算：1277952（768 可一次 2 张）
- 当前正式 run：`runs/v46final_a100_50k`（PID 2705513）
- 实测 step 100：`3992 ms`，loss `5.97212`，PSNR(teacher) `18.34`
- 学生精度：fp32
- 数据根目录顺序固定：
  1. `data/train/proxy_proc`
  2. `data/raw/DIV2K_train_HR`
  3. `data/raw/DAVIS/JPEGImages/480p`
- 数据准备方式：
  - DIV2K 与 DAVIS 480p 由远端 `aria2c` 直接下载并解压
  - `data/proc` 由 `gen_sequences.py` 以 8 个 shard 并行生成 200 组序列
  - `data/train/proxy_proc` 由 `degrade.py` 以 8 个 shard 并行生成，保持串行序号种子

## 6. 启动与监控

启动：

```sh
cd /data2/hezm/klss
bash scripts/start_train.sh
```

监控：

```sh
tail -f /data2/hezm/klss/logs/v46final_a100_50k.log
nvidia-smi -i 6
cat /data2/hezm/klss/runs/v46final_a100_50k/log.csv
```

停止：

```sh
kill "$(cat /data2/hezm/klss/runs/v46final_a100_50k/train.pid)"
```

## 7. 验收顺序

1. `uv` 环境可导入 torch，且只能看到 A100
2. `nr_kernels.so` 以 `sm_80` 构建成功
3. `check_numerics.py --torch` 通过
4. 1 step / 2 图的 smoke test 通过
5. 正式训练启动，日志出现 step、loss，GPU 6 有显存和利用率

## 8. 其他

- 不修改其他用户的进程、环境或 `/data2/hezm` 下无关目录
- Comet 当前启用：项目 `dlss-student`，实验
  `https://www.comet.com/koth-chen/dlss-student/296136fcf1404b7da558f3f24ee30f14`
- Comet key 位于 `code/teacher/.comet_key`（权限 600，不对外分发）
- Comet 指标按 `log_every=100` 步节流上报
- `torch.compile` 于 2026-10-18 在 torch 2.14.0+cu126 上实测：修复
  `row_window` 尾 lane 的假阳性越界后能跑，但首图 warmup 约 12.25 分钟，
  且会按尺寸继续重编译，不具备训练价值；保持 `student_compile=false`
- `shared-routed-moe-v1` 明确拒绝 `bf16`（要求显式 FP32），因此保持
  `student_precision=fp32`
- 教师权重不可提交或打包分发；学生权重的分发条款见 `code/teacher/README.md`
- 若远端代码与本地不一致，以本地同步副本为准，并在日志中记录来源哈希

## 9. 完整数据补齐（2026-10-18）

- 先前 run 只同步了 3 个根，共 8,608 张；v45c 完整数据是 8 个根、30,582 张
- 完整数据位于本机 `/f/yanr/rs`，正在同步到远端 `data/full/`
- 同步由 `remote/sync_full_data.sh` 执行，断线自动重试并续传
- 同步完成后 `remote/launch_when_synced.sh` 会自动：
  1. 停止当前 8,608 张的 fallback run
  2. 启动 `runs/v47_full_320512_50k`
  3. 使用完整八源、50K 步、batch 10、320/512 = 7:3、FP32 + 局部 TF32

## 10. 局部 TF32（待 A/B）

- 新增 `--student-tf32 1`：只让学生 FFN、window/ViT attention、expert GEMM
  使用 TF32
- router 和最终 head 显式保持 FP32，避免路由翻转和输出契约漂移
- teacher 走自研 CUDA 定点链，不受 TF32 影响
- 基准脚本：`remote/bench_precision.sh 0|1`，用相同 seed/data 跑 30 步对比
- 只有 step time 明显下降且 loss/PSNR 可接受时才在正式 run 打开
- 2026-10-18 A/B（100 步）：TF32 comp 约快 5–7%；step 50 PSNR 22.80 vs 22.79，
  step 99 PSNR 19.05 vs 18.85。router/head 仍为 FP32，因此正式完整数据 run 开启
  `student_tf32=true`
- 2026-10-18 数据管线 A/B（pair_fraction=1 压测）：prep=4 的 step 10/20/29
  为 3429/5622/1846ms；prep=8 为 3948/6056/1909ms。prep=8 反而略慢，
  保留 `prep_workers=4`、`teacher_workers=1`
- 2026-10-18 pair 后处理 A/B（pair_fraction=1）：post=1 的 step 10/20 为
  3442/5649ms、wait 1142/1983ms；post=4 为 2916/3585ms、wait 0/0ms，
  loss/PSNR 与串行完全一致。正式 run 使用 `post_workers=4`
- 全量 run 验证：step 200 2678ms/wait 818ms（旧 3550/1568），
  step 300 2596ms/wait 0ms（旧 4064/1506）；同 step 的 loss/PSNR 完全一致

## 11. 2026-10-18 停训

- 用户判定 ViT 架构无效，服务器训练已停止
- 停止进程 PID 3086307；GPU6 已释放（显存回到 5 MiB、利用率 0%）
- 保留 run 产物与日志，不再自动重启：
  - `runs/v47c_full_320512_50k/`
  - `logs/v47c_full_320512_50k.log`
- 数据、权重、代码均未删除

## 12. 新 CNN U-Net MoE8（独立实验）

- 新文件：`code/teacher/nr_cnn_unet.py`、`code/teacher/train_distill_cnn.py`
- 结构：7 级 CNN U-Net，每块共享 3x3 conv + 8 专家 top-1 FFN
- 参数量：62,116,253（62.116M）
- 配置：`config/run_cnn_unet.json`
- 启动：`scripts/start_train_cnn.sh`
- 新 trainer 已加入 teacher/student 流水重叠；30 步 benchmark 的 step
  10/20/29 为 2494/2844/1304ms
- 当前 run：`runs/cnn_unet_moe8_50k_run3`（PID 3404656）
- Comet：`https://www.comet.com/koth-chen/dlss-cnn-unet/13788ee9a6504c2e862ce0a353337d2a`
- 重叠版实测 step 200/300：2306/2673ms（串行版为 3902/4519ms），
  loss/PSNR 与串行版逐位一致
- CNN U-Net compile 实测：30 步 step 10/20/29 为 2399/2747/1306ms，
  eager 为 2494/2844/1304ms；约快 3–4%，warmup 2.16 分钟。
  正式 run5 开启 `student_compile=true`
- 新 trainer 计时：step 10/19 的 `prep/wait/comp` 为 `0/0/2471` 和
  `0/0/1811` ms；数据预取已饱和，瓶颈是学生 GPU 计算。保持
  `prep_workers=4`、`post_workers=4`
- Comet 项目切回 `dlss-student`，避免 dashboard 只显示旧项目时看不到新 run
- 当前 compile 正式 run：`runs/cnn_unet_moe8_50k_run6`（PID 3454419）
- Comet（已切回旧项目）：`https://www.comet.com/koth-chen/dlss-student/f6a8223a04af41018a1759268ef50135`
- compile 实测 step 100：2395ms（eager 2652ms），`prep=0 wait=0 comp=2391`
- 旧串行版 run2 已停止
- 与现有 `train_distill.py` / `student_shared_moe_v1` 完全独立
- 当前正式 run：`runs/v47c_full_320512_50k`（PID 3086307），完整 30,582 张，
  320/512 = 7:3，batch 10，TF32
- Comet：`https://www.comet.com/koth-chen/dlss-student/872bf9c53bbf4f24b9a17ce06b3568a1`
