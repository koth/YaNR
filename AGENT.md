
## 项目
蒸馏出 DLSS5 NR 教师网络(71 块 / 147MB / 512² 113ms)的小模型学生:同几何、同 16 lane
输入契约、同 composite 语义,目标 5× 以上加速。当前进展与架构详见 `teacher/STUDENT.md`;
决策记录(D1–D11,含"为什么不继承 fp8"等关键取舍)见
`openspec/changes/fast-student-distillation/design.md`,任务清单见同目录 `tasks.md`。

## 交流
尽量使用中文交流。

## 服务器 / 训练
- 训练与 GPU 基准都在服务器:`ssh koth@192.168.3.32`(单卡 3090)。网络走代理
  `http://192.168.3.127:7897`(pip `--proxy`,curl `-x`)。
- ssh 用连接复用:ControlMaster / ControlPath=/tmp/ssh_mux_dlss / ControlPersist=20m,
  不要反复开新会话。
- 服务器是 python3.8:pip 用 `python3 -m pip install --user <pkg> --proxy ...`;
  torch 2.4.1+cu121 已装。教师跑 `NR_WEIGHTS=~/klss/models/nr`;CPU 路径设 `NR_DEVICE=cpu`、
  `CUDA_VISIBLE_DEVICES=""`。
- 训练指标上报 CometML(project `dlss-student`),key 在 `teacher/.comet_key`
  (gitignore,勿提交;配置见 train_distill.py 的 comet_key())。

## 目录
- `teacher/` —— 自包含实现:教师(nr_*/run_image/annotate)、学生(nr_student/run_student)、
  训练(train_distill)、成本模型(cost_model/sweep_shapes)、校验与工具。
- `engine/` —— 学生 CPU 推理引擎(C++17 + OpenMP):lane 管线、前向、f16 基元、几何;
  AVX2 数值内核(gemm.cpp 打包 GEMM/exp/int8、attn.cpp 融合注意力)、`gemm_bench` 微基准。
  i9-10850K/8 线程实测 320² 43ms、512² 136ms(fp32;int8 混合质量过线但本机平手)。
  512²/30ms 在本机物理不可达(硬算术见 STUDENT.md §6);口径见 openspec 8.7/8.8。
- `shapes/` —— 学生形状 DSL(student_v0 默认,两个备选)。
- `openspec/` —— 变更提案/任务/规格;完成的任务及时勾选并附数字结论。
- `data/`、`weights/`、`tmp/`、`runs/`、`teacher/.comet_key` 均在 gitignore:
  数据不入库;教师权重/DLL 带 NVIDIA 条款,**绝不随源分发**。

## 工作约定
- 文件改动只用 write/edit 工具;shell 仅用于运行程序、测试、只读查看。
- 数值语义改动必须跑对账回归:`teacher/check_lanes.py`、`check_engine.py --bisect`、
  `check_numerics.py`、`check_fused.py`、`check_model.py`、`test_cost_model.py`;
  新旧实现不一致时先查"语义差异"再改代码(窗注意力掩码维、张量布局、f16 舍入都踩过坑)。
- 确定性优先:数据生成/采样用 `np.random.default_rng([seed, ...])`,同 seed 逐位可复现。
- CPU 引擎改动后跑 `check_engine.py --bisect`(逐级中间量对账,能快速定位首个发散层)。
- CPU 基准必须钉 `OMP_NUM_THREADS=8` 并注明机器负载:10850K 上 20 线程 HT 过订阅会慢 ~8×;
  性能数字用 median,单次 forward 含冷启动无意义。
- 训练实验用 `train_distill.py --size 320,512 --size-weights 0.7,0.3` 这类显式配置,
  起跑前对每种边长做冒烟(场分辨率 != 有效分辨率,曾因此崩过)。

## 常用命令
- 成本记账:`python3 teacher/cost_model.py --shape shapes/student_v0.json --sizes 320,512,768,1080`
- 教师出图:`python3 teacher/run_image.py input.png --width 512 --height 512 -o out/`
- 学生出图:`python3 teacher/run_student.py input.png --width 512 --height 512 -o out/`
- CPU 引擎构建:`cd engine && cmake -B build -S . && cmake --build build -j 8`
- 训练:`python3 teacher/train_distill.py --teacher-weights ~/klss/models/nr --data <roots...> -o runs/vX`
