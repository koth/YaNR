#!/usr/bin/env python3
"""Audit sparse student parameters without importing torch or loading weights.

    python teacher/audit_moe.py weights/student_v44 --verify --output tmp/v44_audit.json

Parameter equivalents are accounting across layers, not image FLOPs or latency.
New shared-routed layouts store shared expert 0 and routed experts 1..16 in
each bank; exported buffers must carry isParameter=false.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


ARCHITECTURE = 'shared-routed-moe-v1'
BANK_NAMES = {'expand', 'narrow', 'contract', 'qkv', 'proj', 'weight'}
TRANSITIONS = {'enc_trans', 'dec_trans', 'vit_in', 'vit_out'}


def _routing(shape):
    """bank 轴数与共享槽数从形状的 routing 段读,不写死 17。

    2026-10-18 去掉共享专家后是 7 个纯路由专家、top-1;(轴数, 共享槽, top_k)。
    """
    r = shape.get('routing') or {}
    return (int(r.get('routed_experts', 7)), int(r.get('shared_experts', 0)),
            int(r.get('top_k', 1)))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def _relative(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError('package path escapes its root: ' + str(relative))
    return path


def _bank_layout(name, shape):
    """Return (bank size, always-on slices, selected slices), or None."""
    parts = name.split('.')
    new = shape.get('architecture') == ARCHITECTURE
    if parts[0] in TRANSITIONS:
        return _routing(shape) if new else None
    if parts[0] in ('enc_blocks', 'dec_blocks') and len(parts) >= 4:
        config = next((lv for lv in shape['levels'] if lv['level'] == parts[1]), None)
        if config is None:
            raise ValueError('tensor references unknown level: ' + name)
        if new and config['kind'] == 'shared_routed_moe':
            return _routing(shape)
        if config['kind'] == 'moe':
            return int(config['experts']), 0, 1
    if parts[0] == 'vit_blocks':
        if new:
            return _routing(shape)
        experts = int(shape['vit'].get('experts', 0))
        if experts >= 2:
            return experts, 0, 1
    return None


def _public_category(name):
    if '.router.' in name:
        return 'router'
    if name.split('.')[0] in TRANSITIONS:
        return 'transition'
    if '.attention.' in name:
        return 'attention_prior_scale'
    if name.startswith(('adapter.', 'head.')):
        return 'input_output'
    return 'residual_merge_other'


def audit_manifest(manifest, shape):
    """Classify every tensor and reject malformed or inconsistent accounting."""
    totals = dict(parameters=0, public_parameters=0, expert_bank_parameters=0,
                  shared_expert_parameters=0, routed_expert_parameters=0,
                  selected_routed_equivalent=0, buffers=0)
    public = {}
    transitions = []
    tensors = []
    names = set()
    for entry in manifest['tensors']:
        name = entry['name']
        if name in names:
            raise ValueError('duplicate tensor: ' + name)
        names.add(name)
        dims = entry['shape']
        if any(type(d) is not int or d <= 0 for d in dims):
            raise ValueError('invalid tensor dimensions: ' + name)
        count = math.prod(dims)
        if entry['dtype'] != 'f32' or entry['byteLength'] != count * 4:
            raise ValueError('invalid f32 tensor size: ' + name)
        record = dict(name=name, shape=dims, elements=count)
        if not entry.get('isParameter', True):
            totals['buffers'] += count
            record['category'] = 'buffer'
        else:
            totals['parameters'] += count
            layout = _bank_layout(name, shape)
            suffix = name.rsplit('.', 1)[-1]
            bank = (layout is not None and
                    ((len(dims) == 3 and suffix in BANK_NAMES) or
                     (name.split('.')[0] in TRANSITIONS and len(dims) == 2 and suffix == 'bias')))
            if bank:
                size, shared, selected = layout
                if dims[0] != size:
                    raise ValueError('expert bank size mismatch: ' + name)
                per_expert = count // size
                totals['expert_bank_parameters'] += count
                totals['shared_expert_parameters'] += per_expert * shared
                totals['routed_expert_parameters'] += per_expert * (size - shared)
                totals['selected_routed_equivalent'] += per_expert * selected
                record.update(category='expert_bank', experts=size,
                              shared_experts=shared, parameters_per_expert=per_expert)
            else:
                category = _public_category(name)
                totals['public_parameters'] += count
                public[category] = public.get(category, 0) + count
                record.update(category='public', public_category=category)
            if name.split('.')[0] in TRANSITIONS and suffix in ('weight', 'bias') and '.router.' not in name:
                transitions.append(record.copy())
        tensors.append(record)

    declared = manifest.get('model', {}).get('params')
    if declared is not None and declared != totals['parameters']:
        raise ValueError('declared model parameter count does not match tensor audit')
    declared = manifest.get('totals', {}).get('paramCount')
    if declared is not None and declared != totals['parameters']:
        raise ValueError('declared total parameter count does not match tensor audit')
    totals['active_parameter_equivalent'] = (totals['public_parameters'] +
                                            totals['shared_expert_parameters'] +
                                            totals['selected_routed_equivalent'])
    fraction = totals['public_parameters'] / totals['parameters'] if totals['parameters'] else 0
    new = shape.get('architecture') == ARCHITECTURE
    budget = dict(applicable=new, target_parameters=80_000_000, tolerance_fraction=0.01,
                  maximum_public_fraction=0.01, actual_public_fraction=fraction)
    if new:
        from moe_shape import ledger
        expected_layout = {e['name']: (e['shape'], e['isParameter']) for e in ledger(shape)}
        actual_layout = {e['name']: (e['shape'], e.get('isParameter', True)) for e in manifest['tensors']}
        expected = ({'enc_trans.' + n + '.' + s for n in ('d0', 'd1', 'd2', 'd3', 'd4') for s in ('weight', 'bias')} |
                    {'dec_trans.' + n + '.' + s for n in ('d0', 'd1', 'd2', 'd3') for s in ('weight', 'bias')} |
                    {n + '.' + s for n in ('vit_in', 'vit_out') for s in ('weight', 'bias')})
        budget.update(total_pass=79_200_000 <= totals['parameters'] <= 80_800_000,
                      public_pass=fraction <= 0.01,
                      layout_pass=expected_layout == actual_layout,
                      transitions_pass=({t['name'] for t in transitions} == expected and
                                        all(t['category'] == 'expert_bank' for t in transitions)))
        budget['pass'] = all(budget[key] for key in ('total_pass', 'public_pass', 'layout_pass', 'transitions_pass'))
    return dict(totals=totals, public_by_category=public, transitions=transitions,
                budget=budget, tensors=tensors,
                interpretation='Parameter equivalent across layers; not whole-image weight usage, MACs, or latency.')


def audit_package(directory, verify=False):
    root = Path(directory).resolve()
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('model', {}).get('kind') != 'student':
        raise ValueError('expected a student package')
    shape_path = _relative(root, manifest['model'].get('shapeFile', 'shape.json'))
    shape = json.loads(shape_path.read_text(encoding='utf-8'))
    report = audit_manifest(manifest, shape)
    stages = {}
    identities = []
    for entry in manifest['stages']:
        path = _relative(root / 'model', entry['file'])
        if entry['id'] in stages:
            raise ValueError('duplicate stage: ' + entry['id'])
        size = path.stat().st_size
        if size != entry['packedByteLength']:
            raise ValueError('stage size mismatch: ' + str(path))
        stages[entry['id']] = size
        identity = dict(file=entry['file'], bytes=size, declared_sha256=entry['sha256'])
        if verify:
            identity['actual_sha256'] = sha256(path)
            if identity['actual_sha256'] != entry['sha256']:
                raise ValueError('stage sha256 mismatch: ' + str(path))
        identities.append(identity)
    for entry in manifest['tensors']:
        size = stages.get(entry['stage'])
        offset = entry['stageOffset']
        if (size is None or type(offset) is not int or offset < 0 or offset % 4 or
                offset + entry['byteLength'] > size):
            raise ValueError('tensor outside stage: ' + entry['name'])
    report['package'] = dict(directory=str(root), version=manifest['model'].get('version'),
                             manifest_sha256=sha256(manifest_path), shape_sha256=sha256(shape_path),
                             stages=identities, stage_hashes_verified=verify)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights')
    parser.add_argument('--verify', action='store_true', help='Hash all binary stages, not only metadata')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit_package(args.weights, args.verify)
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
        print(json.dumps(dict(output=str(args.output), totals=report['totals'], budget=report['budget']), indent=2))
    else:
        print(text, end='')
    return 1 if report['budget'].get('pass') is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
