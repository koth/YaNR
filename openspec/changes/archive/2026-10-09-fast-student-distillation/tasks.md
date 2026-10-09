## 1. 成本模型(cost-estimator)

- [x] 1.1 新建 `teacher/cost_model.py`:教师形状按 nr_network/nr_geometry 布局常量显式建模(通道/hidden/expert 数/分支数/块数),level rows 由 `geometry_from_valid` 推导,输出逐块 GMAC 表(实现注:不做字节反推,manifest 体积交叉核对并入 1.9)
- [x] 1.2 分项公式实现并单测:plain FFN(2·ch·h + 4ch²)、expert FFN(ch²(h/32+1) + ch·h)、split(**6**ch² + 2E·bc·mc,branch/contract + qkv/proj)、ViT(2·ch·ffn + 4ch²)、窗口注意力(128·ch,8×8 槽位)、过渡 gemm/adapter/head 项
- [x] 1.3 重放校验:teacher 形状 512² 总计 **69.57 GMAC**(目标 68.5 ±3%,实差 +1.6%),预测 114.6ms vs 实测 113ms(+1.4%);逐级占比与 design.md 成本地图一致(d4 rows 修正为 320)
- [ ] 1.4 实测校准:用 `teacher/profile_nr.py` 收集 320²/512²/768²/1080p 的分 kernel 时间,拟合每类算子的 G products/s 上限,模型输出 ms 预估(各分辨率误差 ≤10%)
- [x] 1.5 学生形状 DSL:`shapes/*.json` 解析为形状 spec(逐级 channels/hidden/blocks/attn/kind,split 的 branches/bc/mc,ViT channels/ffn/blocks),非法值(非 32 倍数、split 缺字段)显式报错
- [x] 1.6 CLI:`python3 cost_model.py --shape teacher|shapes/*.json --sizes 320,512,768,1080 [--gops N] [--json]`,输出逐级预算表、总计、预测 ms;`shapes/student_v0.json` 实测 **512² 13.69 GMAC → 5.08×**,320² 5.67×(预测门禁 ≥5.0 过)
- [x] 1.7 单测 `teacher/test_cost_model.py`:12 用例全过(分项公式手算对账、teacher 重放 ±3%、各级占比平坦性、宽度单调性、DSL 报错、v0 ≥5× 门禁)
- [x] 1.8 扫描脚本 `teacher/sweep_shapes.py`:宽度 {0.5,0.6,0.75} × 深度 {0.4,0.6,0.8} × 全分辨率策略 {ffn-only, attn-8x8-h32, attn-4x4} × ViT 块数 {2,3,4} = 81 形状,输出 CSV(`tmp/shape_sweep.csv`)+ markdown 报告;**13/81 过 ≥5.0×**,top 均为 w0.5/0.6 × d0.4;cost_model 增加 window_slots(4×4 = 16 槽 = 32·ch)
- [x] 1.9 候选冻结与排序:① `shapes/student_v0.json`(已冻,5.08×,深层层级保宽度换质量)② `student_alt_depth.json`(5.32×,均匀缩宽减深,深层层级欠拟合时换)③ `student_alt_attn4x4.json`(5.44×,D3 备选:full 级 4×4 窗注意力 + d0 减半,质量不达标时启用,需 task 3.7 kernel);预算报告 `tmp/cost_report.md`

## 2. 学生网络骨架(student-network,torch 路径)

- [x] 2.1 新建 `teacher/nr_student.py`:`load_student_shape` + 校验(32 倍数、kind 合法、split 需 E·bc=ch 分组切片约束),默认 `shapes/student_v0.json`
- [x] 2.2 块计划表:每级 enc=blocks//2、dec=其余(full 1+1),phase 用 `WindowPhases.take()` 同规则(full=6,d0..d4=0..4,解码续计);`student_alignment()` 导出 teacher↔student 对齐表
- [x] 2.3 plain FFN 块(`FFNBlock`):expand→SiLU→contract + 残差(aux 缩放),标准 matmul(D11)
- [x] 2.4 `ExpertBlock`(E 全输入→h、h→32 窄支、合并 ch→ch)与 `SplitBlock`(branch、E bc→mc→bc、contract),einsum 分组矩阵乘;修出规格矛盾:split 要求 **E·bc=ch**(教师分组切片语义),三个形状文件与 sweep 规则已对齐
- [x] 2.5 FFN-only 块 = `FFNBlock(attn=False)`:去 qkv/attention/projection,保留残差+aux 语义(full 级即此)
- [x] 2.6 `VitBlock` + `VitAttention`:expand→contract(残差)→qkv→全 token 余弦注意力(列掩码、query √32·scale)→proj(残差);数值自由后不用 expWeight 表/softmax 树
- [x] 2.7 plumbing:box_downsample(2×2 ((a+b)+(c+d))·0.25 越界 0)、UpsampleMerge(最近邻 2×+skip·aux)、PostBlend(aux_pair)、head 32→4、blend_scale 参数
- [x] 2.8 `capture()` 钩子 + `student_alignment()` 对齐表(13 点:s-enc-full..d4 / s-vit / s-dec-d4..d0 / s-head → block-0/4/8/14/22/30/38/47/55/61/65/69/head)
- [x] 2.9 `teacher/run_student.py`:CLI 与 run_image 同语义(input/width/height/seed/style/tone/structure/automask + --shape/--checkpoint/--repeat),同名 PNG 输出 + composite 语义(blend×blend_scale clamp)
- [x] 2.10 smoke 3090 通过:512²/320² 随机权重前向出图,9.29M 参数,head rgb/4 ±0.18、blend 正常,无 NaN;修出 3 个实现 bug(窗索引缺交叉积展开、merge 未 flatten、dec_trans 方向)
- [x] 2.11 序列模式:`run_student.py` 接 `--prev`/`--motion`/`--emit-history`(nr_history,D10 语义),composite 走 temporal blend(blend×blend_scale clamp);3090 两帧实验通过(frame2 含历史正常出图)
- [x] 2.12 基准(`bench_student.py`,3090):eager/compile × f32/f16 四模式 × 三分辨率,**compile+autocast f16 胜出:512² 11.05ms(vs 教师 113ms = 10.2×)、768² 14.18ms(16.3×)、320² 9.69ms(5.9×)**;有效吞吐 1241 GMAC/s@512²(教师 fp8 链的 2×);cost model 用教师校准 607 G/s 偏保守 2×,学生后端预测系数定为 ~1200 G/s

## 3. 学生推理后端(怎么快怎么来,D11)

- [x] 3.1 torch eager 基线:512² 26.6ms(f32)/24.1ms(f16)、320² 21.5ms —— launch/开销瓶颈明显(320²≈512² 早期形态),定位结论:小算子太多,融合是钱
- [x] 3.2 torch.compile(`mode='reduce-overhead'`)计时:eager→compile 提速 **2.1–2.4×**;注意相位参数曾触发 dynamo SymInt 崩溃,已在构造期定死相位修复(图更稳)
- [ ] 3.3 热点手写融合(候选:FFN 块 expand→SiLU→contract、窗注意力 gather→score→softmax→v):compile16 已达 10–16×,**降级为可选优化**(若训练后实测仍差再做)
- [ ] 3.4 窗注意力后端选型:torch gather 参考 vs 自写 kernel,8×8 窗;输出质量守门(与 eager 参考差 ≤1e-3)
- [ ] 3.5 dtype 选型:速度上 **autocast f16 全面胜出**(compile16 vs compile32:+16% @512²、+63% @768²);待训练后补质量守门(eval PSNR 退化 ≤0.1dB,否则 f32 微调),结论写 README
- [ ] 3.6 端到端 3090 计时:学生全图 vs cost model 预测(误差 ≤15%),报告逐 kernel/op 占比,>90% 时间在选定后端
- [ ] 3.7 (条件触发,仅当 D3 备选启用)4×4 窗口注意力(16 槽)实现 + 计时,质量守门同 3.4

## 4. 数据管线(distill-data)

- [x] 4.1 `tools/fetch_data.sh`:DIV2K_train_HR(3.53GB,800 张 2K)+ DAVIS-2017-trainval-Full-Resolution(2.96GB)+ 480p(833MB 可选)全部下载完成,sha256/字节数/URL/时间入 `data/raw/manifest.txt`,断点续传 + `.done` 标记;解压就位(服务器 `~/klss/data/raw` 14GB);注:工作区非 git 仓库,gitignore 项不适用;DIV2K_valid 官方 404,验证集走 train 划 held-out
- [x] 4.2 `teacher/gen_sequences.py`:程序化序列生成(平移/缩放/旋转/粒子/棋盘/流场,fbm 纹理+实心几何体+细线+HUD 面板),1080p、8 帧、200 序列,seed 逐位可复现(已验证);全量已生成 `data/proc/`(1.6GB)
- [x] 4.3 `teacher/degrade.py`:proxy 退化管线(降采样混叠→上采样→unsharp 光晕→模糊→噪声→色带→TAA 式残影),参数逐序列抽取、`--mode random|none|hard`、`degrade_index.jsonl` 记录全参数;`--report` 输出 Sobel 边缘能量与径向 FFT 对比;确定性已验证(同 seed 逐位一致)
- [ ] 4.11 退化参数校准:待真实游戏 proxy 样本或 CG 影片帧到位后,用 `--report` 统计对齐参数范围,回写 degrade.py 默认值与 README
- [x] 4.4 history lane 语义 spike:**结论 = 上帧输出**(D10;docs/network.md lane 表 + docs/frame.md History reconstruction + frame.wgsl 三处互证):存储 composite 码值、`truncate_half` 向零截断、5-tap Catmull-Rom 重投影、centre() 进 lanes 7-9;`teacher/nr_history.py` 实现(恒等运动零误差、整数平移零误差、平滑信号插值误差 0.005)并在 3090 两帧实验验证:无历史 sigmoid 0.15 → 有未补偿历史 0.005(网络正确拒绝错位历史),`--history-source output|proxy` 消融开关就位
- [x] 4.5 `teacher/annotate.py`:在线教师接口 `TeacherAnnotator`(head f32 + 任意捕获点激活,f16 原值)+ 序列 `rollout_sequence`(D10 时序语义,产出历史链)+ `--probe` 捕获点清单(79 点,2.8 对齐表原料)+ `--rollout` 批量输出(head/stored/weight/capture npz + manifest 逐输入 sha256);3090 实测通过(blend_scale 0.7397、历史拒绝行为正确);缓存仅小输出(4.6)
- [ ] 4.6 标注存储预算:**默认在线教师**(融合 kernel 113ms/512² 每 step 现算,不缓存中间激活:full-res 捕获点 18MB/点,5 万样本破 TB),只存源图 crop + seed,输入由 seed 确定性重建;若实测 step 时间超预算,再评估缓存 head(f16 ≈2MB/样本)折中
- [ ] 4.7 噪声策略:每样本师生同 seed;epoch 轮换 seed(≤3 轮)的重标注成本评估并定稿(默认 1 seed)
- [ ] 4.8 crop 与对齐:训练 crop(96×96 / 128×128)按 8×8 窗口对齐 + `nr_geometry` padding 规则一致;写 `test_data_alignment.py` 验证 crop 内特征与整帧一致
- [ ] 4.9 `teacher/report_data.py`:抽样可视化(proxy / 教师输出 / diff×12)、head 分布与 blend 直方图对照 `teacher/samples` 基线、NaN/全零样本过滤
- [ ] 4.10 生成训练集 v1(≥50K 样本对)并 rsync 到服务器(复用 ssh ControlMaster 会话),回传清单校验

## 5. 训练管线(distill-training)

- [ ] 5.1 `teacher/train_distill.py` 骨架:torch + CUDA、AMP bf16、`--config` JSON(dataclass 解析)、单卡入口
- [ ] 5.2 精度对齐:训练 dtype(bf16 AMP + f32 主权重)与推理 dtype(f16)的差距实测;若 eval 退化 >0.1dB,加 f16 前向微调(备选,非默认);D11 后无 QAT
- [ ] 5.3 输出损失:rgb/4 残差 L1 + blend logit SmoothL1 + 合成图(neural)L1,权重可配
- [ ] 5.4 细节损失:拉普拉斯金字塔 L1 + Sobel 边缘 L1(高频加权),权重可配
- [ ] 5.5 特征蒸馏损失:按 `student_alignment()` 逐级 cosine + L2,深层权重低(λ 表写进 config);对齐发生在伪量化后的激活上
- [ ] 5.6 时序损失:相邻帧输出差分 vs 教师差分一致(L1);history rollout 用上帧学生输出(EMA warmup);时序权重默认低(spike 未结论前)
- [x] 5.7 优化器与调度:AdamW(lr 2e-4, wd 0.01)、warmup + cosine、EMA 0.999、grad clip 1.0;损失权重与 lane 采样范围随 config 落盘
- [x] 5.8 日志:csv(`log.csv`)+ 逐 step **CometML** 上报(四类损失分量/逐点特征损失/PSNR/lr/耗时/周期图,project `dlss-student`,key `teacher/.comet_key`;实验 https://www.comet.com/koth-chen/dlss-student/5093241c094341fdaba7eadafe7ba31e)+ 本地拼图(proxy/teacher/student/diff×12)
- [x] 5.8b 训练首轮 NaN 修复:掩码注意力 `masked_fill(-inf)+softmax` 反向在真实数值下产 NaN(梯度解剖定位:attn@v 的 bmm 反向 inf 混合;随机输入复现不了),改 NaN 安全掩码 softmax(finfo.min + where 清零);修复后 0/248 参数梯度非有限,过拟合曲线正常
- [ ] 5.9 checkpoint/resume:每 5K 步存(权重/EMA/optimizer/step/config),`--resume` 全恢复,杀进程重跑零丢失
- [ ] 5.10 `tools/run_train_remote.sh`:服务器 tmux 会话、代理环境变量、日志与 checkpoint 回传、断点续跑;复用 `/tmp/ssh_mux_dlss` ControlMaster
- [x] 5.11 过拟合门禁:3 轮收敛(35.4 → 37.7 → **39.5 peak / 裸权重 8 样本均值 40.50 ≥40 过线**);修掉 3 个问题:warmup 吃满短跑(改 200)、detail loss 写成两图高频能量之和而非场之差(改后 +1.8dB 且不再奖励模糊)、EMA 无暖起(0.999 视界在短跑被半成品污染:32% 质量来自 step1000-2000 → decay 按步数渐升);EMA 口径待 run v1 复验
- [x] 5.12 sanity run:800 张 DIV2K、320²、20K 步,**无 NaN、逐分量下降**(out 0.20→0.07、feature 1.95→~1.1);泛化验收(DAVIS 12 帧未见数据):raw **41.97dB**、EMA **41.96dB** —— EMA 暖起修复生效(修复前 23.9dB);Comet https://www.comet.com/koth-chen/dlss-student/312d8381d76d47e190e528e09b61e76f
- [ ] 5.13 run v1 全量训练(≤72h,30K–50K 样本,单帧为主 + 低权时序),最优 EMA 权重落 `weights/student/v1/`(本地不入库,服务器留存 + 回传 eval 用副本)
- [ ] 5.14 消融 run(各 20K 步):-特征蒸馏 / -时序 / -细节损失 / 历史来源(output vs proxy),输出对比表(PSNR、blend MAE、时序指标)

## 6. 评估与验收(student-eval)

- [x] 6.1 `teacher/eval_quality.py`:学生 vs 教师(PSNR/SSIM)、残差相关系数、blend logit MAE;数据集级聚合(mean/median/min/max)+ 单图明细,json + md 双出口;同一份 features 喂两个网络。冒烟(33K 步 ckpt,DAVIS 2 图@320):vs 教师 47.25dB / SSIM 0.9992,blend_mae 0.25
- [x] 6.2 对输入口径:PSNR(学生,输入) 对教师基线(512² lake = 38.53 dB)的退化 ≤ 1.0 dB;测试集均值同样达标。**v1.1 slim_mixed(60K)实测(10 图同集):512² 退化 +0.05dB ✅、320² +1.10dB ⚠️ 名义超线 0.10dB(非目标口径,用户验收 512²)**;v0 基线同集 −1.42/−2.90dB,容量差为主(见 report_final §5)
- [x] 6.3 时序稳定性:`teacher/eval_temporal.py` 就绪(逐帧独立反馈环 + 块匹配光流一鱼两吃:历史重投影 + warp error;flicker = std(输出差分-输入差分);`--vary-seed` 切换部署口径;`--hist-src teacher` 开环诊断口径)。**v1.1(带 pair 0.3 + temporal 0.5 腿)定性完成:flicker ×2.67@320² / ×2.78@512²(门 ≤1.5×)未过、warp ×0.97 过;开环 ×2.79 排除反馈环,根因 = 学生近似误差逐帧白化(std(Δe)≈0.012 ≈ 误差 RMSE),合成视角对未迁移到真实视频统计;修复 = 真实视频帧对 + 提权重训(~20h),用户决策本轮不跑、照实记录(report_final §4)**
- [ ] 6.4 性能:320²/512²/768²/1080p graph capture 计时(30 次取中位),512² 实测 ≥4.5×(目标 ≥5.0×),报告逐 kernel 占比与预测偏差
- [x] 6.5 视觉对比:`teacher/quad_compare.py` 四联图管线(proxy / neural / diff×12 / blend)出教师-学生并排版,存 `teacher/samples/`(bear512_quad.png 2060×1028、bear320_quad.png + vs_teacher_diff_x12),会话内已展示:教师/学生不可分,误差集中在纹理区
- [x] 6.6 验收汇总 [`teacher/report_final.md`](../../../teacher/report_final.md):v0 形状表、预算 vs 实测、质量表、时序表、消融表、结论与后续(Plan B MoE、低秩、4×4 窗备选)
- [x] 6.7 回归门禁:teacher 校验套件全绿(`check_numerics.py --torch` PASS、`check_fused.py` PASS、`check_model.py` PASS、`check_block0.py` torch vs oracle **0/32 mismatch**)——学生改动未破坏任何教师断言(webgpu vs oracle 32/32 为既有 WebGPU f16 口径差异,与本次改动无关)

## 7. 打包与文档(student-packaging)

- [x] 7.1 `convert_weights.py --shape student`:学生权重 → `manifest.json` + `model/stages/s*.bin`(与教师同格式,含 sha256 与逐 tensor 偏移)。实测:slim_mixed v1.1 → 4,286,790 参数 / 213 张量 / 3 stage / 17.15MB f32,shape.json 内嵌、`model.kind='student'` 自动探测标记(`nr_package.py` 读写两端)
- [x] 7.2 `run_image.py` / `run_nr.py` 接受学生模型目录(形状自动探测或 `--shape student`),输出图注明模型版本。实测:双入口自动探测 OK;PNG tEXt `Model=student-student_slim_mixed-step59999`;run_nr 学生模式退回 eager 计时(窗注意力掩码布尔索引不可捕获 CUDA graph,打印注明);学生前向包 `torch.no_grad()`
- [x] 7.3 打包校验:转换后学生模型加载 → 与训练导出的 torch 输出逐位一致(同输入同权重)。`check_package.py` 实测:213 张量往返逐位一致、前向 `torch.equal` 逐位一致(max|diff| 0.000e+00)、sha256 逐 stage 校验 → **PACKAGE PASS**
- [x] 7.4 `teacher/README.md` 增补:学生结构表(slim_mixed 逐级 + 参数/GMAC)、cost model/sweep 用法、训练/评估/打包命令、复现步骤与硬件要求(训练 GPU/引擎 AVX2+OpenMP/int8 仅 VNNI 划算)
- [x] 7.5 许可与发布声明:学生权重为蒸馏产物不含 NVIDIA 权重、可独立分发;教师权重/DLL 不随源分发(NVIDIA 条款,各自提取);README「Provenance and licensing」节写明
- [x] 7.6 `openspec archive fast-student-distillation` 已执行(validate 全过;26 个 spec delta 落地 `openspec/specs/`,change 归档为 `2026-10-09-fast-student-distillation`);未完成任务(8.7 VNNI 验收、8.10 DML 备胎、5.14 损失消融、6.4 GPU graph capture 计时、flicker 重训路线)在本表随档保留,验收报告见 `teacher/report_final.md`

## 8. CPU 推理引擎(cpu-engine;Windows 消费机 30ms,512² 口径)

- [x] 8.1 `teacher/export_onnx.py`:学生 → ONNX(opset 17,按边长固定形状 + blend_scale 侧车),torch↔ORT **逐位一致**(max diff 0.0;修过 export/check 随机初始化不同步的假失败)
- [x] 8.6 Linux 预演(20 核 AVX2):**现成栈全部不达标**——ORT f32 157ms@320²/525ms@512²、动态 int8 149/486ms、torch eager 127/427ms、inductor 96ms@320²;ORT 逐算子 profile:仅 20% 时间在 GEMM(146 GMAC/s,不差),27% 是碎 einsum(注意力)、37% 是 Where/Mul/Gather/Scatter 搬运 → **结论:8.9 手写融合是主路径不是备选**
- [x] 8.2 int8 量化阶梯:动态 PTQ 已通但无效(瓶颈不在权重 GEMM);**8.9 融合后重测完成(AVX2)**:int8 前向 27.1/85.0ms(320²/512²,平手/略慢,无 VNNI),合成图 65.4/65.7dB 无损,`NR_INT8=1` opt-in;静态 PTQ/QAT 按需(建议 VNNI 验收后再定)
- [x] 8.3 `engine/` C++ 输入管线:hash_uniform/gaussian3/center_proxy/镜像 padding/条件 lane 移植完成,`check_lanes.py` 对账 **PASS**(max f16 步距 1、越界 0,512/320 双尺寸 + 历史路径);抓到 f16 进位 OR-vs-add bug(148 万元素稀疏错位的根因)
- [x] 8.4 C++ 推理宿主:`cpu_engine` 前向本体完成且 **与 torch 数值一致**(320/512 逐级 max ~5e-7,head 120dB,check_engine --bisect 全链绿);修复 4 个语义级 bug(窗注意力查询维掩码是训练语义、ViT 池化输入、专家/分支 `[E,in,out]` 布局、f16 进位)。composite + PNG 输出 + 统计段补齐:`--proxy/--png/--blend-png`,零依赖 stored-deflate PNG 编码器(png_write.h),口径同 run_image(neural = clamp(proxy + rgb/4)、blend = clamp(sigmoid·blend_scale));v1 ckpt 实测出图干净、统计正常
- [x] 8.5 CMake + MSVC(win-x64)构建支持(编译旗标按编译器分派 /O2 /arch:AVX2 vs -O3 -mavx2;-mfma)+ 一键基准脚本 `engine/bench.sh`(Linux,实测)/ `engine/bench.ps1`(Windows):lane 构造 / 前向 / composite 分段计时,30 次取中位。lane 构造按行并行(逐位不变,check_lanes 照绿):**10.9 → 1.4ms**;320² 端到端 lane 1.4 + 前向 36.2 + composite 1.4 = **~39.4ms**
- [ ] 8.7 Windows 实测验收:512² 端到端 ≤30ms(中位,报 CPU 型号/线程数)、320² ≤15ms;硬件口径:VNNI 级(Intel 12 代+/Zen4)为达标线,AVX2-only 机器出对照数据;未达标输出瓶颈分段 + 降档建议
- [x] 8.8 质量验收:**实测通过** —— int8 引擎 vs f32 学生合成图 PSNR 65.4dB(320²)/65.7dB(512²),退化 ≈0.00dB ≤0.3dB,且远优于 f32 学生对教师口径;head 逐通道 max|diff| rgb 2.3-3.2e-3 / logit 2.6e-2、rgb rmse 3.6e-4,超 2e-3 逐元素线的集中在 logit 通道,合成图无感(check_engine 逐元素线仅作引擎对账参考)
- [x] 8.9 **主路径** 手写融合 CPU 引擎:窗序布局(免 gather/scatter,同 CUDA 窗 kernel 的 field addressing)、融合 FFN(expand→SiLU→contract→residual)、融合注意力(余弦归一→scores→掩码 softmax→V)、int8 GEMM(i16 展开 vpmaddwd,K≥256 阈值)全部落地,对账 5e-6/head 120dB;**实测(slim_mixed 512²,AVX2 8 线程):gemm 50.1 + attn 15.1 + other 18.6 = 83.9ms**(目标 22-34+2-5ms 为 30ms 口径的 VNNI 级预算,AVX2 物理不可达,差距移交 8.7;STUDENT.md §6 硬算术)
- [ ] 8.10 (备胎)DirectML/核显路径:同一 ONNX 走 DML 后端实测 512²,作为消费机无独显场景的第二答案
