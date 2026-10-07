"""The temporal history: what lanes 7-9 see and how it is stored (openspec task 4.4).

spike 结论(见 design.md D10;证据 docs/network.md 的 lane 表、docs/frame.md
"History reconstruction"、shaders/frame.wgsl 的 input_features/compose/truncate_half):

  * lanes 7-9 是**上帧输出** —— composite 码值 neural = clamp(proxy_code + head_rgb/4),
    经 temporal blend(权重 clamp(sigmoid(head3) * blend_scale, 0, 1));不含 style/tone/显示变换。
  * 存储走 truncate_half:f16 **向零截断**(不是四舍五入 —— 反馈环里四舍五入会逐帧单向漂移)。
  * 下一帧在运动重投影位置(current + motion,uv、y 向下)用 5-tap Catmull-Rom 采样
    (双线性折叠技巧,clamp-to-edge;普通双线性会逐帧融化细节)。
  * 采样结果走与 proxy 相同的 centre() 进 lanes 7-9;无历史(首帧/无效运动)时
    lanes 7-9 = 当前 proxy 的拷贝,composite 不混合。

依赖:numpy。
"""
import numpy as np


def truncate_half(values):
    """f16 向零截断(frame.wgsl truncate_half 的逐位移植)。

    输入 float32/float64,返回 float32,值在 f16 网格上且向零取整。
    NaN -> qNaN,溢出 -> ±inf,|v| < 2^-24(半格下溢) -> ±0。
    """
    v = np.asarray(values, dtype=np.float32)
    bits = v.view(np.uint32)
    sign = (bits >> np.uint32(16)) & np.uint32(0x8000)
    exponent = ((bits >> np.uint32(23)) & np.uint32(0xff)).astype(np.int32)
    mantissa = bits & np.uint32(0x7fffff)

    half_exponent = exponent - 112
    normal_bits = (sign | ((half_exponent.astype(np.uint32) & np.uint32(0x1f)) << np.uint32(10))
                   | (mantissa >> np.uint32(13)))
    sub_shift = np.clip(14 - half_exponent, 0, 31).astype(np.uint32)
    sub_bits = sign | ((mantissa | np.uint32(0x800000)) >> sub_shift)
    inf_bits = sign | np.uint32(0x7c00)
    nan_bits = sign | np.uint32(0x7e00)

    out = np.where(exponent == 0xff, np.where(mantissa != 0, nan_bits, inf_bits),
                   np.where(half_exponent >= 31, inf_bits,
                            np.where(half_exponent < -10, sign,
                                     np.where(half_exponent <= 0, sub_bits, normal_bits))))
    return out.astype(np.uint16).view(np.float16).astype(np.float32)


def _bilinear(prev, u, v):
    """frame.wgsl history_bilinear:uv*size-0.5 起点的双线性,clamp-to-edge。"""
    h, w = prev.shape[:2]
    px = u * w - 0.5
    py = v * h - 0.5
    x0 = np.floor(px)
    y0 = np.floor(py)
    fx = (px - x0).astype(np.float32)[..., None]
    fy = (py - y0).astype(np.float32)[..., None]
    xi = x0.astype(np.int64)
    yi = y0.astype(np.int64)
    x0c = np.clip(xi, 0, w - 1)
    x1c = np.clip(xi + 1, 0, w - 1)
    y0c = np.clip(yi, 0, h - 1)
    y1c = np.clip(yi + 1, 0, h - 1)
    top = prev[y0c, x0c] * (1.0 - fx) + prev[y0c, x1c] * fx
    bottom = prev[y1c, x0c] * (1.0 - fx) + prev[y1c, x1c] * fx
    return top * (1.0 - fy) + bottom * fy


def _catmull_weights(pos):
    """单轴 Catmull-Rom 权重(frame.wgsl reprojected_history 的折叠形式)。"""
    base = np.floor(pos - 0.5) + 0.5
    f = np.clip(pos - base, 0.0, 1.0).astype(np.float32)
    square = f * f
    cube = f * square
    w0 = square - 0.5 * (f + cube)
    w1 = (cube * 1.5 - square * 2.5) + 1.0
    w3 = (cube - square) * 0.5
    w2 = (1.0 - w0) - w1 - w3
    middle = w1 + w2                                   # ∈ [1, 1.375],永不为 0
    return base, w0, w2, w3, middle


def reproject_history(prev, motion):
    """把上帧存储历史 prev [h, w, 3] 按运动场 motion [h, w, 2](uv 偏移,y 向下)重投影。

    5-tap Catmull-Rom(与 frame.wgsl 同一折叠技巧、同一 clamp 语义)。
    恒等运动逐像素复原 prev;整数像素平移等于 clamp 边界的平移拷贝。
    """
    prev = np.asarray(prev, dtype=np.float32)
    motion = np.asarray(motion, dtype=np.float32)
    h, w = prev.shape[:2]
    if motion.shape[:2] != (h, w):
        raise ValueError(f'motion shape {motion.shape} does not match history {(h, w)}')
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    pos_x = (motion[..., 0] + (xx + 0.5) / w) * w
    pos_y = (motion[..., 1] + (yy + 0.5) / h) * h

    base_x, w0x, w2x, w3x, midx = _catmull_weights(pos_x)
    base_y, w0y, w2y, w3y, midy = _catmull_weights(pos_y)

    def texel(p, size):
        return np.clip(p, 0.5, size - 0.5) / size

    low_x = texel(base_x - 1.0, w)
    center_x = texel(base_x + w2x / midx, w)
    high_x = texel(base_x + 2.0, w)
    low_y = texel(base_y - 1.0, h)
    center_y = texel(base_y + w2y / midy, h)
    high_y = texel(base_y + 2.0, h)

    a = w0x * midy
    b = w0y * midx
    c = midx * midy
    d = w3y * midx
    e = w3x * midy
    value = (_bilinear(prev, low_x, center_y) * a[..., None]
             + _bilinear(prev, center_x, low_y) * b[..., None]
             + _bilinear(prev, center_x, center_y) * c[..., None]
             + _bilinear(prev, center_x, high_y) * d[..., None]
             + _bilinear(prev, high_x, center_y) * e[..., None])
    return value / (a + b + c + d + e)[..., None]


def store_history(neural):
    """composite 码值 -> 下帧可读的历史存储(frame.wgsl:截断而非四舍五入)。"""
    return truncate_half(neural)


def history_no_motion(prev):
    """恒等重投影的便捷入口(静态内容/无运动向量时)。"""
    h, w = prev.shape[:2]
    return reproject_history(prev, np.zeros((h, w, 2), np.float32))
