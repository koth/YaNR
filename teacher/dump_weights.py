#!/usr/bin/env python3
"""Dump student weights + block config for the C++ engine (openspec tasks 8.4/8.9).

输出一对文件:
  <prefix>.bin   全部浮点张量的 f32 小端裸数据(state_dict 顺序,确定性)
  <prefix>.json  张量清单(name -> shape/offset/dtype)+ 块配置(kind/hidden/expert/attn/shift)

    python3 dump_weights.py --shape ../shapes/student_v0.json --checkpoint runs/v1/ckpt.pt \\
        --size 512 -o engine/weights/student512
(不带 --checkpoint 则导出 seed=0 的随机初始化权重,用于引擎对账。)
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nr_student import StudentNetwork, load_student_shape              # noqa: E402
from nr_geometry import geometry_from_valid                            # noqa: E402


def block_meta(b):
    tail = getattr(b, 'tail', None)
    meta = {'kind': b.__class__.__name__,
            'attn': tail is not None or b.__class__.__name__ == 'VitBlock'}
    if b.__class__.__name__ in ('FFNBlock', 'VitBlock'):
        meta['hidden'] = b.expand.out_features
    elif b.__class__.__name__ == 'ExpertBlock':
        meta['experts'] = b.experts
        meta['hidden'] = b.expand.shape[-1]
        meta['narrow'] = b.narrow.shape[-1]
    elif b.__class__.__name__ == 'SplitBlock':
        meta['branches'] = b.branches
        meta['bc'] = b.bc
        meta['mc'] = b.mc
    if tail is not None:
        meta['window'] = tail.attention.window
        meta['shift'] = [tail.attention.shift_x, tail.attention.shift_y]
    else:
        meta['window'] = 0
        meta['shift'] = [0, 0]
    return meta


def main():
    parser = argparse.ArgumentParser(description='dump student weights for the C++ engine')
    parser.add_argument('--shape', required=True)
    parser.add_argument('--checkpoint', help='train_distill ckpt.pt (omit for seed-0 random init)')
    parser.add_argument('--size', type=int, default=512, help='deploy edge length (recorded only)')
    parser.add_argument('-o', '--out', required=True, help='output prefix')
    args = parser.parse_args()

    torch.manual_seed(0)
    g = geometry_from_valid(args.size, args.size)
    shape = load_student_shape(args.shape)
    model = StudentNetwork(shape, g).eval()
    state = model.state_dict()
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state = ckpt.get('ema', ckpt.get('model', ckpt))       # EMA 优先(部署口径)
        model.load_state_dict(state)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bin_path = args.out + '.bin'
    tensors = {}
    offset = 0
    with open(bin_path, 'wb') as fh:
        for name, tensor in state.items():
            arr = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
            raw = arr.tobytes()
            tensors[name] = {'shape': list(arr.shape), 'offset': offset,
                             'dtype': 'f32', 'bytes': len(raw)}
            fh.write(raw)
            offset += len(raw)

    config = {'size': args.size, 'blend_scale': float(model.blend_scale.detach()),
              'shape_file': args.shape, 'checkpoint': args.checkpoint,
              'geometry': {'full': [g['full_width'], g['full_height']],
                           'levels': [[l['width'], l['height']] for l in g['levels']],
                           'vit_tokens': g['vit_tokens'],
                           'padded_vit_tokens': g['padded_vit_tokens']},
              'plan': model.plan}
    blocks = {}
    for side in ('enc_blocks', 'dec_blocks'):
        for level, module_list in getattr(model, side).items():
            for i, b in enumerate(module_list):
                blocks[f'{side}.{level}.{i}'] = block_meta(b)
    for i, b in enumerate(model.vit_blocks):
        blocks[f'vit_blocks.{i}'] = block_meta(b)
    config['blocks'] = blocks

    json_path = args.out + '.json'
    with open(json_path, 'w') as fh:
        json.dump({'bin': os.path.basename(bin_path), 'tensors': tensors,
                   'config': config}, fh, indent=1, sort_keys=True)

    # 行式索引(.idx):C++ 引擎免 JSON 解析直接读。
    idx_path = args.out + '.idx'
    with open(idx_path, 'w') as fh:
        fh.write(f"BIN {os.path.basename(bin_path)}\n")
        fh.write(f"GEOM {args.size} {g['full_width']} {g['full_height']} "
                 + ' '.join(f"{l['width']} {l['height']}" for l in g['levels'])
                 + f" {g['vit_tokens']} {g['padded_vit_tokens']}\n")
        fh.write(f"BLEND_SCALE {float(model.blend_scale.detach())!r}\n")
        for name, t in tensors.items():
            fh.write('T ' + name + ' ' + str(len(t['shape'])) + ' '
                     + ' '.join(str(d) for d in t['shape']) + ' '
                     + f"{t['offset']} {t['bytes']}\n")
        for key, b in blocks.items():
            fh.write('BLOCK ' + key + ' ' + b['kind'].lower().replace('block', '')
                     + f" {int(b['attn'])} {b.get('hidden', 0)} {b.get('experts', 0)} "
                     + f"{b.get('narrow', 0)} {b.get('branches', 0)} {b.get('bc', 0)} "
                     + f"{b.get('mc', 0)} {b.get('window', 0)} {b['shift'][0]} {b['shift'][1]}\n")
    print(f'wrote {bin_path} ({offset / 1e6:.2f} MB) + {json_path} + {idx_path}  ({len(tensors)} tensors)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
