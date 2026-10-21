"""把 CNN U-Net 的 checkpoint 打成与其它学生包同构的格式。

只读 state_dict,不实例化 CNNUNet —— blend_scale 本身就在 state_dict 里,
所以打包不需要模型对象,也就不会碰到 nr_package._block_order() 里
enc_blocks/dec_blocks/vit_blocks 那套 Swin/ViT 专有假设。

参数前缀 -> block 编号(其余 head/stem/merge 等记 -1):
  enc.<level>.<i>.*   enc_proj.<level>.*   merge.<level>.*   dec.<level>.<i>.*

    python3 make_cnn_unet_package.py --shape ../shapes/cnn_unet_moe8.json \
        --checkpoint weights/cnn_unet_moe8/cnn_unet_moe8_step17999.pt -o <包目录>
"""
import argparse
import hashlib
import json
import os
import re

import numpy as np
import torch

ORDER = ('full', 'd0', 'd1', 'd2', 'd3', 'd4', 'd5')
LEVEL_BLOCKS = re.compile(r'^(enc|dec)\.([a-z0-9]+)\.(\d+)\.')
LEVEL_ONLY = re.compile(r'^(enc_proj|merge)\.([a-z0-9]+)\.')
STAGES = 3


def block_ids(names):
    """按 (侧, 级, 序号) 汇总出稳定的 block 编号;级顺序走 ORDER。"""
    keys = set()
    for name in names:
        m = LEVEL_BLOCKS.match(name)
        if m:
            keys.add((m.group(1), m.group(2), int(m.group(3))))
            continue
        m = LEVEL_ONLY.match(name)
        if m:
            keys.add((m.group(1), m.group(2), 0))
    order = []
    for side in ('enc', 'enc_proj', 'merge', 'dec'):
        for level in ORDER:
            for key in sorted(k for k in keys if k[0] == side and k[1] == level):
                order.append(f'{key[0]}.{key[1]}.{key[2]}')
    return {key: idx for idx, key in enumerate(order)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--state', choices=('ema', 'model'), default='ema',
                    help='取 EMA 还是当前模型权重(默认 ema;checkpoint 无 ema 时用 model)')
    ap.add_argument('-o', '--out', required=True)
    a = ap.parse_args()

    with open(a.shape) as fh:
        shape_text = fh.read()
    shape = json.loads(shape_text)
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if a.state == 'ema':
        # 导出包里的 *_full.pt 同时带 model 与 ema;EMA 通常更稳,作为默认。
        if 'ema' not in ckpt:
            raise SystemExit('checkpoint has no ema; use --state model')
        state = ckpt['ema']
    else:
        state = ckpt.get('model', ckpt)
    step = int(ckpt.get('step', -1))

    blocks = block_ids(state)
    params = set(state)

    records = []
    for name, tensor in state.items():
        arr = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
        m = LEVEL_BLOCKS.match(name) or LEVEL_ONLY.match(name)
        key = f'{m.group(1)}.{m.group(2)}.{int(m.group(3)) if m.re is LEVEL_BLOCKS else 0}' \
            if m else None
        block = blocks.get(key, -1) if key else -1
        parameter = name.split('.', 1)[1] if block >= 0 else name
        records.append((name, block, parameter, arr))

    total = sum(arr.nbytes for *_, arr in records)
    per_stage = (total + STAGES - 1) // STAGES
    stage_list, tensors = [], []
    current = None
    for name, block, parameter, arr in records:
        raw = arr.tobytes()
        if current is None or (len(current[1]) >= per_stage and len(stage_list) < STAGES):
            current = (str(len(stage_list)), bytearray())
            stage_list.append(current)
        stage_id, buffer = current
        tensors.append({'name': name, 'block': block, 'layer': 0, 'parameter': parameter,
                        'stage': stage_id, 'stageOffset': len(buffer),
                        'byteLength': len(raw), 'shape': list(arr.shape), 'dtype': 'f32',
                        'isParameter': name in params})
        buffer += raw

    os.makedirs(os.path.join(a.out, 'model', 'stages'), exist_ok=True)
    manifest_stages = []
    for stage_id, buffer in stage_list:
        rel = f'stages/s{stage_id}.bin'
        with open(os.path.join(a.out, 'model', rel), 'wb') as handle:
            handle.write(buffer)
        manifest_stages.append({'id': stage_id, 'file': rel,
                                'packedByteLength': len(buffer),
                                'sha256': hashlib.sha256(bytes(buffer)).hexdigest()})

    with open(os.path.join(a.out, 'shape.json'), 'w') as handle:
        handle.write(shape_text)
    shape_name = os.path.splitext(os.path.basename(a.shape))[0]
    blend = float(np.asarray(state['blend_scale'].detach().cpu()).reshape(-1)[0]) \
        if 'blend_scale' in state else 1.0
    manifest = {
        'totals': {'blockCount': len(blocks), 'tensorCount': len(tensors),
                   'paramCount': int(sum(int(np.prod(t['shape'])) for t in tensors
                                         if t['isParameter']))},
        'model': {'kind': 'student', 'shape': shape_name, 'shapeFile': 'shape.json',
                  'step': step,
                  'params': int(sum(int(np.prod(t['shape'])) for t in tensors
                                    if t['isParameter'])),
                  'blendScale': blend,
                  'version': f'student-{shape_name}-step{step}'},
        'stages': manifest_stages,
        'tensors': tensors,
    }
    with open(os.path.join(a.out, 'manifest.json'), 'w') as handle:
        json.dump(manifest, handle, indent=1)

    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(a.out) for f in fs)
    print(f'wrote {a.out}  step {step}  blocks {len(blocks)}  tensors {len(tensors)}  '
          f'{size / 1e6:.1f} MB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
