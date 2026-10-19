"""逐比特比对:同一 checkpoint、同一输入下,两个代码版本的前向输出是否完全相同。

评测指标只到小数点后 2-3 位,不足以证明"clamp 只影响被 mask 的尾 lane"。
这里直接对输出张量的字节做哈希。
"""
import hashlib
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())                  # 由调用方 cd 到目标代码目录
from nr_geometry import geometry_from_valid      # noqa: E402
from nr_student import StudentNetwork, load_student_shape  # noqa: E402
from run_image import build_features             # noqa: E402

device = 'cuda'
w, h = 320, 512
g = geometry_from_valid(w, h)
shape = load_student_shape('/home/koth/klss/shapes/student_shared_moe_v1.json')
image = '/f/yanr/rs/gameplay/Ghost_of_Tsushima___UOheOYV-H2U_frames000008006.png'
proxy = np.asarray(Image.open(image).convert('RGB').resize((w, h), Image.LANCZOS),
                   dtype=np.float32) / 255.0
feats = torch.from_numpy(build_features(proxy, g, 7, 2.0, 1.0, 1.0, -1.0, True)).to(device)

torch.manual_seed(0)
student = StudentNetwork(shape, g).to(device).eval()
ckpt = torch.load('/home/koth/keep_tmp/v45c_44k/ckpt.pt', map_location='cpu', weights_only=False)
student.load_state_dict(ckpt.get('ema', ckpt.get('model')))
with torch.no_grad():
    out = student(feats).float().cpu()

raw = out.numpy().tobytes()
print(f'module   : {__import__("nr_shared_moe").__file__}')
print(f'shape    : {tuple(out.shape)}')
print(f'md5      : {hashlib.md5(raw).hexdigest()}')
print(f'sum/abs  : {out.sum().item():.10f} / {out.abs().sum().item():.10f}')
print(f'first6   : {out.flatten()[:6].tolist()}')
