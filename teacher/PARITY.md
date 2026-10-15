# 教师逐位对账

算子数值单测、融合前后相等和整网与原生实现逐位一致，是不同的验证。
`fixtures/numerics.bin` 只覆盖数值基元；`check_fused.py` 检查融合内核及图调用的部分契约。
它们通过后，仍需用相同输入、相同权重跑独立实现，并比较整网的实际输出。

## OpenDLSS-NR 的标准

依据上游 [数值规范](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1/docs/numerics.md)
和 [WebGPU 校验说明](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1/ports/browser-webgpu/README.md)：

| 检查 | 通过标准 |
| --- | --- |
| 70 个块输出（block 0～69）及 5 个编码器转换，共 75 个边界 | E4M3 字节完全相同 |
| 最终 head | f32 位模式完全相同 |
| 浮点合成图 | RGB half 发布后的位模式完全相同 |
| 旧版 RGBA8 截图 | 每通道最多相差 1 个码值，单独报告为 within tolerance |

`+0` 和 `-0` 不同也算失败。浮点比较不能用 `allclose`、PSNR 或简单的数值相等代替；
NaN 也只允许相同位模式。旧版截图的 1/255 容差不能用于 head、中间边界或浮点合成图。

fixture 的 `manifest.json` 声明 `checks`（`boundaries`、`head`、`output`）及对应参考文件。
边界测试必须覆盖全部 75 个位置；缺失项需要显式注明省略原因。
原生 Vulkan 的规范还要求生产调度、逐次屏障调度和抓取调度的 head 相同，并重复运行确认稳定。
`verify` 用于定位 block 0 的算子差异，真正的验收命令是 `parity`。

## 跑上游原生 captures

原生 captures **没有随上游源码发布**。必须另行提供；不能拿本地教师生成的参考文件证明
本地教师与 NVIDIA 直接一致。

在上游 `ports/browser-webgpu` 目录下运行（需要 Node、Chrome 和支持 WebGPU 的 GPU）：

```powershell
$env:NR_WEIGHTS = 'D:\work\YaNR\weights\teacher'
$env:NR_FIXTURES = 'D:\captures'  # 包含原生 nr512、nr768 两个 fixture 目录
node tools/check.mjs
```

它运行算术 self-test 和两个互补的整网 fixture：

- `nr512`：75 个边界逐字节一致，旧版合成截图在 1 个码值内。
- `nr768`：head 和浮点合成输出逐位一致。

也可以在编译好的 Vulkan 版本运行：

```powershell
.\build\dlss5vk.exe parity --model D:\work\YaNR\weights\teacher --fixture D:\captures\nr512
.\build\dlss5vk.exe parity --model D:\work\YaNR\weights\teacher --fixture D:\captures\nr768
```

仅运行 self-test、`verify` 或只比较 PNG，均不构成上述完整验收。
权重版本、输入尺寸、padded field、16-lane features、随机种子、控制参数、mask 与历史帧都应固定；
定位前向计算时优先直接共享 `features.f32.bin`，避免把预处理差异误判成网络计算差异。

## 2026-10-15 本地实测与修复

上游版本：`9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1`。
使用 RTX 3080 上的独立 WebGPU 实现，对照本地 CUDA 教师，加载 `weights/teacher` 的真实权重。
两侧共享完全相同的 f32 features；原图尺寸，参数为 style=2、tone=1、structure=1、skin=-1、
automask=1，seed=12345，首帧历史为当前图。

发现 CUDA 和 Triton 的 `fp8_gemm_partitioned` 图接口漏用了 A 输入的 `packedInputIndex`
通道重排。普通 GEMM 和 torch 图路径有此重排。错误影响 ViT 的分块 GEMM 及相应解码投影。
t6 首个分歧在 block 31；修复前 76 项中 36 项逐位一致、40 项失败。

CUDA 修复后的整网比较：

| 图片 | 有效尺寸 | 边界 | head 的 f32 元素数 | 位模式差异 |
| --- | --- | --- | --- | --- |
| t6 | 1080×669 | 75/75 逐字节一致 | 3,244,032 | 0 |
| t7 | 1280×720 | 75/75 逐字节一致 | 4,128,768 | 0 |

每张图片均为 76 项 bit-exact、0 项失败、0 项容差通过。
新增三种 partition 的图接口回归在修复前失败、修复后通过。
Triton 包装层同步修复，但 Windows 环境未安装 Triton，尚未执行其 GPU 回归。

本地要求的 `check_lanes.py`（含历史帧）、`check_engine.py --bisect`、
`check_numerics.py --torch`、`check_fused.py`、`check_model.py` 和 `test_cost_model.py` 均通过。
上游 WebGPU self-test 也通过。

这证明了该输入下的 **本地 CUDA 网络前向与上游 WebGPU 前向一致**。
参考文件由本地教师生成并交给未修改的上游 `runParity` 检查器；它们不是 NVIDIA captures。
本次 `checks` 为 `boundaries`、`head`，没有把 PNG 合成或不同后端的预处理记作逐位验收通过。
本地图片入口直接合成 PNG，上游原生浮点输出还包含 half 截断发布，两者必须另外核对。

额外将上游从 t6 proxy 生成的 features 输入修复后的本地教师，3,244,032 个 head 值也全部逐位一致。
Python 与 WebGPU 各自生成 features 时，噪声 lane 0～2 分别有 1402、1417、1439 个位模式差异
（每 lane 共 811,008 个值，最大绝对差 0.001953125）；其余 13 个 lane 全部逐位相同。
因此“网络前向对齐”不能表述成“各自从 PNG 开始的完整入口已逐位对齐”。

此前基于错误 CUDA 路径导出的教师图像、PSNR 和纹理结论需要重测。
如果学生训练时在线使用了该 CUDA/Triton 路径或缓存了其输出，训练目标也可能受影响；
需核对实际训练代码版本、后端及缓存来源，不能仅凭学生版本名认定是否受影响。

t1～t6 已按上述相同参数、原始尺寸重出 `teacher/samples/tN/teatcher.png`。
旧图备份在 `tmp/teacher_before_partition_swizzle_fix/tN/teatcher.png`。
下表是输出相对原图的 PSNR，仅反映改动幅度，不是逐位对账或画质优劣的判据：

| 图片 | 修复前 dB | 修复后 dB |
| --- | --- | --- |
| t1 | 39.19 | 25.89 |
| t2 | 41.42 | 27.41 |
| t3 | 36.95 | 28.32 |
| t4 | 41.39 | 31.98 |
| t5 | 39.24 | 26.95 |
| t6 | 36.52 | 27.14 |
