#!/usr/bin/env python3
# Convert the DLL's packed WEIGHTS_HT resource into a model directory this runtime loads.
#
# The packed resource comes from MLX-DLSS's extractor:
#   mlxdlss-weights extract nvngx_dlssnr.dll dlssnr-weights-packed.safetensors
# It holds the DLL's 153 named weight records (blockN.layerM.<parameter>) as the raw packed bytes - the
# exact layout docs/weights.md describes and nr_model.py unpacks (including the 16-byte zero alignment
# trailers the DLL appends to some records: they are part of the record and harmless to carry along).
#
# The output is manifest.json + stages/s*.bin with real sha256s, loadable via NR_WEIGHTS=<dir>.
#
#   python3 convert_weights.py weights/extracted/dlssnr-weights-packed.safetensors -o weights/nr
#   python3 convert_weights.py --shape student runs/v1/ckpt.pt -o weights/student_mixed   # 7.1 学生打包

import argparse
import hashlib
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NAME = re.compile(r'^block(\d+)\.layer(\d+)\.(.+)$')


def read_packed(path):
    with open(path, 'rb') as handle:
        header_len = struct.unpack('<Q', handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        base = 8 + header_len
        payload = handle.read()
    records = []
    for name, info in header.items():
        if name == '__metadata__':
            continue
        lo, hi = info['data_offsets']
        records.append((name, payload[lo:hi]))
    return header.get('__metadata__', {}), records


def convert_student(ckpt_path, out, stages, size=512, shape_json=None):
    """学生权重打包(7.1):train_distill ckpt.pt -> manifest.json + model/stages/s*.bin。

    与教师同格式(sha256 + 逐 tensor 偏移),张量为 f32 裸数据;形状自动探测
    (ckpt config 的 shape 路径,可用 --shape-json 覆盖),shape.json 随包内嵌。
    """
    import torch
    from nr_geometry import geometry_from_valid
    from nr_package import write_package
    from nr_student import StudentNetwork, load_student_shape

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('ema', ckpt.get('model', ckpt))       # EMA 优先(部署口径,同 dump_weights)
    cfg = ckpt.get('config', {}) or {}
    step = int(ckpt.get('step', -1))
    shape_path = shape_json or cfg.get('shape')
    if not shape_path or not os.path.isfile(shape_path):
        raise SystemExit(f'shape json not found: {shape_path!r} (pass --shape-json)')

    shape = load_student_shape(shape_path)
    g = geometry_from_valid(size, size)
    torch.manual_seed(0)
    model = StudentNetwork(shape, g).eval()
    model.load_state_dict(state)                            # strict:形状与权重必须一致

    manifest = write_package(model, state, out, shape_path, step=step, stages=stages,
                             extra={'trainConfig': {k: v for k, v in cfg.items()
                                                    if k in ('size', 'size_weights', 'steps',
                                                             'pair_fraction', 'loss_temporal')}})
    meta = manifest['model']
    print(f"student package: {meta['version']}  {meta['params']:,} params  "
          f"{sum(t['byteLength'] for t in manifest['tensors']) / 1e6:.2f} MB f32  "
          f"{len(manifest['tensors'])} tensors -> {out}")
    for entry in manifest['stages']:
        print(f"  stage {entry['id']}: {entry['packedByteLength']} bytes  sha256 {entry['sha256'][:16]}…")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packed', help='dlssnr-weights-packed.safetensors (the WEIGHTS_HT resource); '
                                      'with --shape student: train_distill ckpt.pt')
    parser.add_argument('-o', '--out', required=True, help='output model directory')
    parser.add_argument('--stages', type=int, default=3, help='number of stage files to split into')
    parser.add_argument('--template', default=None,
                        help='an existing manifest.json to cross-check record sizes against')
    parser.add_argument('--shape', default=None,
                        help="'student': 打包学生 ckpt(7.1);教师打包不需要")
    parser.add_argument('--shape-json', default=None,
                        help='学生形状 DSL 路径(默认从 ckpt config 自动探测)')
    parser.add_argument('--size', type=int, default=512,
                        help='学生打包时用于结构校验的边长(仅校验,默认 512)')
    args = parser.parse_args()

    if args.shape == 'student':
        return convert_student(args.packed, args.out, args.stages,
                               size=args.size, shape_json=args.shape_json)
    if args.shape is not None:
        raise SystemExit(f'--shape {args.shape!r} 未支持(教师打包不需要;学生用 --shape student)')

    meta, records = read_packed(args.packed)
    print(f"packed resource: {len(records)} records, "
          f"{sum(len(b) for _, b in records)} bytes, format {meta.get('format')}")

    if args.template:
        with open(args.template) as handle:
            template = json.load(handle)
        expected = {t['name']: t['byteLength'] for t in template['tensors']}
        diffs = [(n, len(b), expected.get(n)) for n, b in records if expected.get(n) != len(b)]
        for name, real, templ in diffs:
            print(f'  size differs from template: {name}: {real} vs {templ} '
                  f'({"+16 zero trailer" if templ is not None and real - templ == 16 else "?"})')
        if len(diffs) != sum(1 for n, b in records if n not in expected):
            print(f'  ({len(records) - len(expected)} records not in template)')

    parsed = []
    for name, blob in records:
        match = NAME.match(name)
        if not match:
            raise ValueError(f'unexpected record name {name}')
        parsed.append((name, int(match.group(1)), int(match.group(2)), match.group(3), blob))

    # Pack the records into stage files in order, balancing the totals.
    total = sum(len(b) for *_, b in parsed)
    per_stage = (total + args.stages - 1) // args.stages
    stages = []
    tensors = []
    current = None
    for name, block, layer, parameter, blob in parsed:
        if current is None or (len(current[1]) >= per_stage and len(stages) < args.stages):
            current = (str(len(stages)), bytearray())
            stages.append(current)
        stage_id, buffer = current
        tensors.append({
            'name': name, 'block': block, 'layer': layer, 'parameter': parameter,
            'stage': stage_id, 'stageOffset': len(buffer), 'byteLength': len(blob),
        })
        buffer += blob

    # The loader reads stage files relative to <dir>/model/ (docs/weights.md).
    os.makedirs(os.path.join(args.out, 'model', 'stages'), exist_ok=True)
    manifest_stages = []
    for stage_id, buffer in stages:
        rel = f'stages/s{stage_id}.bin'
        path = os.path.join(args.out, 'model', rel)
        with open(path, 'wb') as handle:
            handle.write(buffer)
        manifest_stages.append({
            'id': stage_id, 'file': rel, 'packedByteLength': len(buffer),
            'sha256': hashlib.sha256(bytes(buffer)).hexdigest(),
        })
        print(f"  stage {stage_id}: {len(buffer)} bytes -> {rel}")

    manifest = {
        'totals': {'blockCount': 71},
        'stages': manifest_stages,
        'tensors': tensors,
    }
    with open(os.path.join(args.out, 'manifest.json'), 'w') as handle:
        json.dump(manifest, handle, indent=1)
    print(f"wrote {os.path.join(args.out, 'manifest.json')}: {len(tensors)} tensors, "
          f"{sum(t['byteLength'] for t in tensors)} bytes total")


if __name__ == '__main__':
    main()
