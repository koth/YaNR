# v45 训练的分块与显存

2026-10-17，本机 RTX 3080 10GB，Torch 2.6.0+cu124，FP32、TF32 off、确定性模式，CPU 8 线程。
桌面与其他应用占用约 2GB 显存；未关闭用户应用。没有可用的 3090，24GB 配置是容量建议，尚未在 24GB 卡实测。

## `batch_rows` 的选择

`batch=8` 是每次 optimizer 更新的有效 batch；`batch_rows` 限制一次学生前向/反向的实际 field 行数，
桶内通过梯度累积完成 batch=8。320 的 field 为 320×320；512 的 field 为 **576×512**，不能按 512² 计算。
每张图至少单独运行一次，预算小于一张时不会切割图片。时序 pair 的两次学生前向共用这个分块大小。

| 配置 | 学生行预算 | 同尺寸桶内 320 上限 | 512 上限 | 教师行预算 |
|---|---:|---:|---:|---:|
| 原配置 | 262144 | 2 | 1 | 原实现没有限制 |
| 本机 10GB 的低显存配置 | 262144 | 2 | 1 | 294912（320 两张 / 512 一张） |
| 本机 10GB 的折中起点 | **307200** | **3** | **1** | **294912** |
| 根目录 v45 的 24GB 建议 | **1179648** | **8**（受 batch 限制） | **4** | **589824**（320 五张 / 512 两张） |

同一步不同尺寸分桶，实际每桶可能更小。路由 padding、通道宽度、attention 和 optimizer 状态也占显存，
行预算不是完整显存预算。如果另有 **5GB 数据常驻 GPU**，也必须计入；数据文件在磁盘上占 5GB 则不等于占 5GB 显存。
本次使用 `teacher/samples` 冒烟，未拿到原训练数据，不能声称复现或排除了该数据路径的 OOM。

本机完整配置的 4 步短跑，保留 batch=8、320/512 混合、在线教师、纹理/特征/时序损失，
只将教师路径改成本地、禁用 Comet/可视化/定时 checkpoint，并因 Windows 无可用 Triton 禁用 CUDA Inductor：

| 学生预算 / 教师预算 | 4 步总墙钟（含初始化、最终保存） | PyTorch allocated 峰值 | reserved 峰值 |
|---|---:|---:|---:|
| 262144 / 294912 | 41.79 s | 5.54 GiB | 6.60 GiB |
| 307200 / 294912 | 40.97 s | 6.12 GiB | 7.09 GiB |
| 409600 / 294912 | 72.66 s（前一轮 68.13 s） | 7.09 GiB | 8.02 GiB |

短跑不是统计稳定的端到端基准；409600 接近本机显存上限，出现明显的步骤延迟尖峰。
本机可从 307200 起步（最后一步 4.27 s，262144 为 4.92 s），显存更紧则用 262144；不建议继续增大。
24GB 的建议值需要在目标卡上验证，不能把本机学生单独运行的吞吐外推为完整训练速度。
旧提交 `796ff4f` 的同配置完整短跑在约 169 秒内未完成第一步，手动停止；没有计入有效速度对比。

10GB 卡运行时覆盖根配置：

```bash
python teacher/train_distill.py --config run_v45_cfg.json --data <数据目录> -o runs/v45-local --batch-rows 307200 --teacher-batch-rows 294912
```

Windows 本机测试还需在配置中设 `student_compile=false`。显存仍紧时可加 `--no-teacher-overlap`，让教师和学生 GPU 段串行，CPU 准备仍预取。
`teacher_workers` 当前为兼容配置项；GPU 教师由一个执行器串行调度，增加它不会产生更多 GPU 标注并发。

## 已做的优化

- 深层/ViT 不再给每个非空专家固定分配至少 4096 行，而按输入行数选 8～4096 的 tile；保留所有 token、路由选择和 gate。
- 路由固定 16 槽计数，移除动态 `repeat_interleave` 的额外同步；分组布局在同一块内复用。
- 已知排列及其逆排列用确定的 gather 完成反向；专家 tile 的梯度按连续分段求和，避免通用索引反向的排序/散射开销。
- 缓存静态窗索引，避免每次前向构造几何及 GPU 布尔索引的尺寸同步；清理前一次学生捕获引用。
- 学生和教师分别按实际 field 行数分块；教师只保留所需的 12 个捕获点，原先会保留 79 个。
- 非日志步跳过路由健康指标、梯度范数和损失明细的 GPU→CPU 传输；日志步批量传输可合并的标量。
- 新结构 CUDA 默认 fused AdamW，EMA 用 foreach 发射；resume 在 CPU 读取 checkpoint 后按需恢复，释放整个 checkpoint 引用。

新结构仍使用 FP32，没有靠 BF16、丢 token、改损失权重或强制配平提速。
分块、GEMM tile 和 fused AdamW 会改变浮点归约/更新的舍入，不能承诺与旧版训练轨迹逐位一致。
`optimizer_fused=false` 可保留原 AdamW 实现。已有 `ema_every` 字段在本次前后都未参与调度；EMA 仍每步更新，未悄悄改变其训练语义。
完整 optimizer step 边界的 checkpoint 允许调整 `batch_rows` 后继续训练；正在积累微批的 checkpoint 仍要求行预算一致，避免跳过或重复样本。

## 学生单独运行的基准

`bench_moe_training.py` 包含前向、head MSE、反向、AdamW、controller 和 EMA，不含在线教师与完整蒸馏损失。
每组预热 2 次，取后续中位数；320 单张优化后及 512 单张旧版采样 3 次，其他表内组采样 5 次。
不同 batch 的每一行都单独做一次 optimizer 更新；表中每张耗时不是相同有效 batch 的训练加速比。

| 输入 / 微批 | `796ff4f` ms | 优化后 ms | 优化后每张 ms | allocated 峰值（旧→新） |
|---|---:|---:|---:|---:|
| 320 / 1 | 654.10 | 479.73 | 479.73 | 3.84 → 2.47 GiB |
| 512 / 1 | 667.25 | 584.71 | 584.71 | 5.69 → 4.19 GiB |
| 320 / 4 | — | 750.41 | 187.60 | 5.80 GiB |
| 512 / 2 | — | 1095.42 | 547.71 | 6.95 GiB |

```bash
python teacher/bench_moe_training.py --size 320 --batch 1 --baseline-ref 796ff4f --output tmp/train_perf/baseline320.json
python teacher/bench_moe_training.py --size 320 --batch 4 --output tmp/train_perf/current320b4.json
python teacher/bench_moe_training.py --size 512 --batch 2 --output tmp/train_perf/current512b2.json
```

CUDA 测试设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；各 GPU 测试顺序执行。原始报告保存在 `tmp/train_perf/`，
可入库的采样记录见 [v45_training_20261017.json](benchmarks/v45_training_20261017.json)。
本机只验证了 eager 性能和 Dynamo eager 的动态前向/反向，未验证 Linux CUDA Inductor 的编译耗时、显存及吞吐。

## 数值验证

分组投影与独立逐行 oracle 对比输出、输入/权重/bias 梯度，覆盖空专家、全热专家、多 tile 和重复确定性。
静态窗缓存与旧实现比较 CPU/CUDA、奇数尺寸、多 batch、四种相位的前向及反向，逐位一致。
教师捕获裁剪的 head 和全部所需特征逐位一致。Python/C++ shared MoE 在 320²、512²、320×512、137×201 对账，所有路由一致，head 最大绝对误差 3.76e-6。
同时执行既有 lanes、engine bisect、numerics、fused、model、cost_model 和损失批量回归；本次没有启动长训或作出纹理质量结论。
