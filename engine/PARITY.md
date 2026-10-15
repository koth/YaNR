# C++ / Python 学生模型对账

2026-10-14，本地源码 HEAD `c7e05a6`，包含当前未提交改动及本文记录的 half 舍入修复。
验收对象为 `weights/student_v3`、`weights/student_v4`。
参考为现有 uv `.venv` 中的 PyTorch CPU，C++ 为 MSVC Release CPU / CUDA FP32，
CUDA 显卡为 RTX 3080。未启用 INT8 / TF32。

**v3 的 9 组、v4 默认参数的 2 组均通过数值对账。**
v4 仅允许测试 `style=64, tone=1, structure=1, skin=1, automask=on`，
seed 为 CLI 默认的 12345，采用静态图片输入；尺寸覆盖 320×320、512×512。
回归脚本对 v4 固定使用这两组用例，v3 保留控制参数覆盖。

## 复现

在项目根目录运行，使用现有 uv 环境；不传 `--weights` 时默认测试 v3 / v4：

```powershell
$env:PYTHONUTF8 = '1'
.venv/Scripts/python.exe teacher/check_cpp.py --cuda --bisect --report tmp/cpp_parity/v3_v4.json

# 只验收 v4：仅运行默认参数的 320×320、512×512
.venv/Scripts/python.exe teacher/check_cpp.py --cuda --bisect `
    --weights weights/student_v4 --report tmp/cpp_parity/v4_defaults_only.json

# 两套权重的默认参数 320×320
.venv/Scripts/python.exe teacher/check_cpp.py --cuda --quick

# v3 的 style 边界覆盖
.venv/Scripts/python.exe teacher/check_cpp.py --cuda --weights weights/student_v3 `
    --case style0-default-controls --case style127-default-controls
```

`check_cpp.py`、`check_gpu.py` 均对 `student_v4` 限定默认参数。
通过 `--case` 选择其他用例时，v4 会跳过；若没有可执行的用例，立即报错，
不会把空测试记为通过。

脚本直接读取权重包，Python 加载时验证 stage SHA256。图片先统一为同一份 PNG，
排除 PIL Lanczos / WIC Fant 缩放差异。比较全部 16 个输入 lane、完整 FP32 head、
有效区域 PNG、逐级/逐块捕获，以及 CUDA eager / Graph 输出。
另用 Python 独立构造的 lanes 再跑一次模型，避免只检查 C++ 导出的输入。

验收沿用 `check_engine.py` 的 head 最大绝对误差 `2e-3`，PNG 最大误差 `1/255`。
中间量误差用于定位，独立记录，不作为最终 head 通过的依据。
失败会写入 JSON 并返回非零退出码。

## 实测结果

输入为 `teacher/samples/lake.png`。

| 权重 | 验收范围 | head/PNG 同时通过 | CPU/Python 最大 head 误差 | CUDA/Python 最大 head 误差 | PNG 最大通道误差 |
|---|---|---:|---:|---:|---:|
| student_v3 | 默认参数、style 边界、控制项、尺寸、history、half 边界 | 9/9 | 1.717e-5 | 2.289e-5 | 1/255 |
| student_v4 | 仅默认参数，320×320 / 512×512 | 2/2 | 1.240e-5 | 1.192e-5 | 1/255 |

这 11 组输入 lanes 均与 Python **逐位一致**；CUDA eager / Graph head 均逐位一致。
C++ / Python 的 head 是容差内一致，PNG 并非逐字节一致。

v4 补测 `teacher/samples/t2/t2.png`、`teacher/samples/t6/t6.png`，
均保持默认参数、320×320，也通过，最大 head 误差 `1.240e-5`。
v4 的验收结论仅针对上述约定参数。

v3 的 9 组明细：

- 默认参数 320×320、512×512。
- 320×320，只将 style 分别改为 0、127，其余控制项保持默认。
- 关闭 automask：320×320、style 0、tone 0.25、structure 0.75、seed 0。
- 矩形：320×512、style 127、tone/structure 0、skin 0.25、seed 42。
- 非整齐尺寸：137×201、style 64、tone 1、structure 0.35、skin 跟随 structure。
- history：默认 320×320，历史 proxy 为当前图横向移动 7 像素。
- half 舍入边界：默认 320×320，tone/skin 为 `0.5002441406250001`，
  structure 为 `0.5002441406249999`。

未单列时 seed 为 12345、automask 为 on。v3 的中间 MoE 激活仍可能超出绝对阈值，
本报告只对最终 head / PNG 作通过判定，不声称全部中间激活或专家选择逐位一致。

本地原始记录：v4 为 `tmp/cpp_parity/v4_defaults_only.json`、
`v4_t2.json`、`v4_t6.json`；v3 为 `v3_v4_details.json`、
`v3_v4_styles.json` 中的 student_v3 条目。生成物不入库。

## 已修复：double 到 half 的二次舍入

原 C++ 噪声和控制值使用 `double -> float -> half`，Python 使用 `double -> half`。
接近 half 中点时，先转 float 会改变最后一次 ties-to-even 的方向。
320×320、seed 12345 的噪声曾有 15 个元素不同，最大相差一个 half 刻度。

现在增加直接 f64-to-f16 RNE 转换，噪声和 double 控制值均使用它；RGB proxy 的三次
float/half 舍入保持原语义。`kernel_check` 穷举所有有限正 half 的相邻中点及两侧
double，覆盖正负号、次正规数、溢出和特殊值。本轮所测输入 lanes 已逐位一致。

## 校验范围和限制

- `check_lanes.py`（含 history）、`check_engine.py --bisect`、CTest 两项、
  `check_numerics.py` 的 oracle 对账、`test_cost_model.py` 12 项通过。
- `check_numerics.py --torch` 和 `check_fused.py` 的教师 CUDA 核需要 CUDA 版
  PyTorch，当前 CPU 版环境无法运行；`check_model.py` 缺少教师模型目录。
  学生包仍由本轮实际加载与 SHA256 校验覆盖，以上教师检查不记为通过。
- Python 图片 CLI 仍使用旧参数默认值；对账必须显式指定同一组参数。
  对原图分别使用 WIC / PIL 缩放，也会引入独立的像素差异。
- v3 history case 检查历史 lane 与网络前向；C++ 图片 CLI 是静态 composite，
  不覆盖应用层运动重投影、历史输出混合或 `infer_bgra` 的整段适配器。
