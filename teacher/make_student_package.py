"""把训练 checkpoint 打成学生包(manifest.json + model/stages/s*.bin + shape.json),
供 C++ 引擎按 --weights 加载,从而能在任意尺寸下跑(包不含几何,尺寸运行时给)。

必须带 torch,所以在服务器上跑;产物是普通目录,拷到本地即可。

    python3 make_student_package.py --shape ../shapes/student_shared_moe_v1.json \
        --checkpoint /home/koth/keep_tmp/v45d_56k/ckpt.pt -o /tmp/student_v45d_56k
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_geometry import geometry_from_valid                       # noqa: E402
from nr_package import write_package                              # noqa: E402
from nr_student import StudentNetwork, load_student_shape         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--size', type=int, default=512,
                    help='仅用于构造网络(几何不进包),默认 512')
    ap.add_argument('-o', '--out', required=True)
    a = ap.parse_args()

    shape = load_student_shape(a.shape)
    g = geometry_from_valid(a.size, a.size)
    model = StudentNetwork(shape, g)
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt.get('ema', ckpt.get('model'))
    model.load_state_dict(state)
    state = {k: v.detach().float().cpu() for k, v in state.items()}

    os.makedirs(a.out, exist_ok=True)
    write_package(model, state, a.out, a.shape, step=int(ckpt.get('step', -1)),
                  extra={'trainConfig': {'checkpoint': os.path.basename(a.checkpoint)}})
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(a.out) for f in fs)
    print(f'wrote {a.out}  step {ckpt.get("step")}  {total / 1e6:.1f} MB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
