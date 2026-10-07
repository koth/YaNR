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

import argparse
import hashlib
import json
import os
import re
import struct

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packed', help='dlssnr-weights-packed.safetensors (the WEIGHTS_HT resource)')
    parser.add_argument('-o', '--out', required=True, help='output model directory')
    parser.add_argument('--stages', type=int, default=3, help='number of stage files to split into')
    parser.add_argument('--template', default=None,
                        help='an existing manifest.json to cross-check record sizes against')
    args = parser.parse_args()

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
