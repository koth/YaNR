"""实验 C:学生误差的频谱分布。

用拉普拉斯金字塔把 (学生−教师) 的残差按尺度拆开,看误差集中在低频(糊)还是
高频(缺纹理)。每一带给出 amp(学生/教师的幅度比)、corr(相关) 和
err/teacher = 1 + amp^2 - 2*amp*corr(1.0 表示"什么都没做")。

真实数据口径:`loss = clip(proxy + rgb/4)`。
    CUDA_VISIBLE_DEVICES=0 python3 probes/error_spectrum.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path('/home/koth/klss')
sys.path.insert(0, str(ROOT / 'teacher'))

from annotate import TeacherAnnotator              # noqa: E402
from nr_cnn_unet_fast import CNNUNetFast, load_fast_shape  # noqa: E402
from nr_geometry import geometry_from_valid        # noqa: E402
from run_image import build_features               # noqa: E402
from single_frame_data import read_rgb, RESAMPLE   # noqa: E402

W, H, LEVELS = 512, 320, 4
IMAGES = ['teacher/samples/t1/t1.png', 'teacher/samples/t2/t2.png',
          'teacher/samples/t3/t3.png', 'teacher/samples/t5/t5.png',
          'teacher/samples/t7/t7.jpg']
CKPT = '/f/yanr/keep_tmp/run_cnn_fast_1/ckpt.pt'
SHAPE = ROOT / 'shapes/cnn_unet_fast.json'


def pyramid(image):
    """返回各尺度带 + 最低频残差。image: HxWx3 float。"""
    bands, current = [], image.astype(np.float32)
    for _ in range(LEVELS):
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
        bands.append(current - up)
        current = down
    bands.append(current)
    return bands


def stats(s, t):
    amp = float(s.std() / max(t.std(), 1e-9))
    sc, tc = s - s.mean(), t - t.mean()
    corr = float((sc * tc).mean() / max(sc.std() * tc.std(), 1e-12))
    return amp, corr, 1.0 + amp * amp - 2.0 * amp * corr


def main():
    g = geometry_from_valid(W, H)
    ann = TeacherAnnotator(str(ROOT / 'models/nr'), g)
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = CNNUNetFast(load_fast_shape(SHAPE), g)
    model.load_state_dict(ckpt['ema'] if 'ema' in ckpt else ckpt['model'])
    model.eval().switch_to_deploy().cuda()
    print('student step', ckpt.get('step'), flush=True)

    accum = []
    psnrs = []
    for path in IMAGES:
        image = read_rgb(ROOT / path)
        proxy = np.asarray(image.resize((W, H), RESAMPLE), dtype=np.float32) / 255
        feats = build_features(proxy, g, 91, 2., 1., 1., -1., True)

        ann.g = ann.network.g = g
        with torch.inference_mode():
            tres = ann.annotate(feats.reshape(-1, 16), capture=(), batch=1)
        t_head = tres['head'].reshape(H, W, 4)
        model.g = g
        with torch.inference_mode():
            s_head = model(torch.from_numpy(feats).cuda()).cpu().numpy().reshape(H, W, 4)

        def composite(head):
            return np.clip(proxy + head[:, :, :3] / 4.0, 0.0, 1.0)

        sc_, tc_ = composite(s_head), composite(t_head)
        mse = float(np.mean((sc_ - tc_) ** 2))
        psnrs.append(10 * np.log10(1 / max(mse, 1e-12)))
        # 关心的是"改动量"而不是图像本身:分别减掉输入再做金字塔。
        s_delta, t_delta = sc_ - proxy, tc_ - proxy
        sb_, tb_ = pyramid(s_delta), pyramid(t_delta)
        accum.append([stats(a, b) for a, b in zip(sb_, tb_)])

    arr = np.array(accum)          # (images, levels, 3)
    print(f'整体 s-vs-t: {np.mean(psnrs):.2f} dB  (每张 {[round(p,2) for p in psnrs]})')
    labels = ['1-2px', '2-4px', '4-8px', '8-16px', '残余(低频)']
    print(f'{"尺度":<12}{"amp":>8}{"corr":>8}{"err/teacher":>13}')
    for i, label in enumerate(labels):
        amp, corr, ratio = arr[:, i, :].mean(0)
        print(f'{label:<12}{amp:>8.2f}{corr:>8.2f}{ratio:>13.2f}')

    # 误差能量在各个尺度的占比
    total = 0.0
    shares = []
    for i in range(len(labels)):
        share = float((arr[:, i, 2]).mean())
        shares.append(share)
        total += share
    print('误差份额:', {labels[i]: f'{100 * shares[i] / total:.0f}%'
                       for i in range(len(labels))})
    Path('/tmp/error_spectrum.json').write_text(json.dumps(
        dict(psnr=psnrs, levels=[dict(amp=float(a), corr=float(c), ratio=float(r))
                                 for a, c, r in arr.mean(0)]), indent=1))


if __name__ == '__main__':
    main()
