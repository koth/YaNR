"""结构消融:ViT(全局通路)到底在不在干活?

ViT 位于编码器瓶颈:enc 到 d4 后下采样到 d5,过 vit_blocks,再解码回 full。
用 forward hook 做三种条件的对比(同一批裁剪、同一权重):

  baseline   原样
  no-vit     让每个 vit block 变成恒等映射 —— 只去掉"全局混合",保留 d5 的内容
  vit-mean   把 vit 输出替换成逐样本空间均值 —— 只留全局直流,毁掉空间变化

分别按尺度算与教师改动的相关。若 no-vit 与 baseline 几乎相同,说明 ViT 没起作用;
若 vit-mean 明显变差,说明 ViT 传递的空间信息是必要的。
"""
import glob
import os
import random
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/home/koth/klss/teacher')
from nr_geometry import geometry_from_valid                  # noqa: E402
from nr_model import Model                                    # noqa: E402
from nr_network import Network                                # noqa: E402
from nr_student import StudentNetwork, load_student_shape     # noqa: E402
from run_image import build_features                          # noqa: E402

W, H = 320, 512
CKPT = '/home/koth/keep_tmp/v45d_56k/ckpt.pt'
SHAPE = '/home/koth/klss/shapes/student_shared_moe_v1.json'
RECIPE = dict(seed=12345, style=2.0, tone=1.0, structure=1.0, skin=-1.0, automask=True)
ROOTS = ['/f/yanr/rs/DIV2K_train_HR', '/f/yanr/rs/Flickr2K', '/f/yanr/rs/gameplay',
         '/f/yanr/rs/portraits', '/f/yanr/rs/rgb']
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def pyramid(a):
    levels = [a]
    while min(levels[-1].shape[:2]) >= 8:
        h, w, c = levels[-1].shape
        h2, w2 = h // 2, w // 2
        levels.append(levels[-1][:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, c).mean((1, 3)))
    bands = []
    for i in range(len(levels) - 1):
        up = np.repeat(np.repeat(levels[i + 1], 2, 0), 2, 1)[:levels[i].shape[0],
                                                             :levels[i].shape[1]]
        bands.append(levels[i] - up)
    bands.append(levels[-1])
    return bands


def band_corr(dt, dp):
    return [float(np.corrcoef(bt.ravel(), bp.ravel())[0, 1])
            for bt, bp in zip(pyramid(dt), pyramid(dp))]


def main():
    files = []
    for root in ROOTS:
        files += [f for f in glob.glob(root + '/**/*.*', recursive=True)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.seed(20261007)
    picked = random.sample(files, N)

    g = geometry_from_valid(W, H)
    teacher = Network(Model('cuda').load('/home/koth/klss/models/nr'), g)
    torch.manual_seed(0)
    student = StudentNetwork(load_student_shape(SHAPE), g).to('cuda').eval()
    student.load_state_dict(torch.load(CKPT, map_location='cpu',
                                       weights_only=False).get('ema'))

    handles = []

    def clear():
        for h in handles:
            h.remove()
        handles.clear()

    def set_no_vit():
        """每个 vit block 变成恒等:输出原样传回输入。"""
        clear()
        for block in student.vit_blocks:
            handles.append(block.register_forward_hook(lambda m, i, o: i[0]))

    def set_vit_mean():
        """最后一个 vit block 的输出换成逐样本空间均值(batch=1 时即全体均值)。"""
        clear()
        def hook(m, i, o):
            return o.mean(0, keepdim=True).expand_as(o)
        handles.append(student.vit_blocks[-1].register_forward_hook(hook))

    conds = [('baseline', clear), ('no-vit', set_no_vit), ('vit-mean', set_vit_mean)]
    acc = {name: [] for name, _ in conds}
    rms = {name: [] for name, _ in conds}
    labels = []

    for k, path in enumerate(picked):
        img = Image.open(path).convert('RGB')
        if img.width < W or img.height < H:
            img = img.resize((max(W, img.width), max(H, img.height)), Image.LANCZOS)
        x = random.randint(0, img.width - W)
        y = random.randint(0, img.height - H)
        proxy = np.asarray(img.crop((x, y, x + W, y + H)), dtype=np.float32) / 255.0
        feats = torch.from_numpy(build_features(
            proxy, g, RECIPE['seed'], RECIPE['style'], RECIPE['tone'],
            RECIPE['structure'], RECIPE['skin'], RECIPE['automask'])).to('cuda')
        with torch.no_grad():
            head = teacher.record(feats).float().cpu().numpy().reshape(
                g['full_height'], g['full_width'], 4)
        t = head[:H, :W, :3] / 4.0

        for name, setup in conds:
            setup()
            with torch.no_grad():
                h = student(feats).float().cpu().numpy().reshape(
                    g['full_height'], g['full_width'], 4)
            s = h[:H, :W, :3] / 4.0
            acc[name].append(band_corr(t, s))
            rms[name].append(float(s.std() * 255))
            if not labels:
                labels = [f'{2 ** i}-{2 ** (i + 1)}px'
                          for i in range(len(acc[name][0]) - 1)] + ['残余']
        clear()
        del feats
        torch.cuda.empty_cache()
        if (k + 1) % 5 == 0:
            print(f'  {k + 1}/{len(picked)}', flush=True)

    print(f'\n裁剪 {len(picked)} 张   输出 rms: ' +
          '  '.join(f'{n}={np.mean(v):.2f}/255' for n, v in rms.items()))
    print(f'{"尺度":<10}' + ''.join(f'{n:>12}' for n, _ in conds))
    for i, lab in enumerate(labels):
        print(f'{lab:<10}' + ''.join(
            f'{np.nanmean([r[i] for r in acc[n]]):12.2f}' for n, _ in conds))


if __name__ == '__main__':
    main()
