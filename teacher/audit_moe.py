#!/usr/bin/env python3
"""Audit sparse student parameters without importing torch or loading weights.

    python teacher/audit_moe.py weights/student_v44 --verify --output tmp/v44_audit.json

Parameter equivalents are accounting across layers, not image FLOPs or latency.
Supports routed banks, per-expert CNN Linear layers and packed CNN banks.
Routing counts come from the shape; exported buffers carry isParameter=false.
The historical 80M / 1% public prototype budget is opt-in (--prototype-budget).
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


ARCHITECTURE = 'shared-routed-moe-v1'
BANK_NAMES = {'expand', 'narrow', 'contract', 'qkv', 'proj', 'weight'}
TRANSITIONS = {'enc_trans', 'dec_trans', 'vit_in', 'vit_out'}
CNN_ARCHITECTURES = {'cnn_unet_moe8', 'cnn-single-moe16-v1', 'cnn-unet-fast8-v1'}


def _routing(shape):
    """bank 轴数与共享槽数从形状的 routing 段读,不写死 17。

    2026-10-18 去掉共享专家后是 7 个纯路由专家、top-1;(轴数, 共享槽, top_k)。
    """
    r = shape.get('routing') or {}
    routed, shared, selected = (int(r.get('routed_experts', 7)), int(r.get('shared_experts', 0)),
                                int(r.get('top_k', 1)))
    if routed < 1 or shared < 0 or not 1 <= selected <= routed:
        raise ValueError('invalid expert routing counts')
    return routed + shared, shared, selected


def _cnn_bank(name, dims, shape, groups):
    """Recognize only expert weights/biases; never treat the router as a bank."""
    if shape.get('architecture') not in CNN_ARCHITECTURES:
        return None
    parts = name.split('.')
    if len(parts) < 5 or parts[0] not in ('enc', 'dec') or parts[3] != 'ffn':
        return None
    if parts[4] == 'router':
        return None
    prefix = '.'.join(parts[:4])
    blocks = shape['blocks_enc' if parts[0] == 'enc' else 'blocks_dec']
    if (parts[1] not in blocks or not parts[2].isdigit() or
            str(int(parts[2])) != parts[2] or int(parts[2]) >= blocks[parts[1]]):
        raise ValueError('unknown CNN expert block: ' + name)
    experts = int(shape['experts'])
    if experts < 1:
        raise ValueError('invalid CNN expert count')
    channels, hidden = shape['channels'][parts[1]], shape['expert_hidden'][parts[1]]
    packed = shape['architecture'] == 'cnn-unet-fast8-v1'
    if packed:
        expected = {'w1': [experts, channels, hidden], 'b1': [experts, hidden],
                    'w2': [experts, hidden, channels], 'b2': [experts, channels]}
        if len(parts) != 5 or parts[4] not in expected or dims != expected[parts[4]]:
            raise ValueError('invalid packed CNN expert tensor: ' + name)
        key = parts[4]
        required = set(expected)
        record = dict(experts=experts, shared_experts=0,
                      parameters_per_expert=math.prod(dims) // experts, representation='packed')
    else:
        if (len(parts) != 8 or parts[4] != 'bank' or not parts[5].isdigit() or
                str(int(parts[5])) != parts[5] or not 0 <= int(parts[5]) < experts):
            raise ValueError('invalid CNN expert index: ' + name)
        expected = {('0', 'weight'): [hidden, channels], ('0', 'bias'): [hidden],
                    ('2', 'weight'): [channels, hidden], ('2', 'bias'): [channels]}
        if (parts[6], parts[7]) not in expected or dims != expected[parts[6], parts[7]]:
            raise ValueError('invalid CNN expert tensor dimensions: ' + name)
        key = tuple(parts[5:])
        required = {(str(e), layer, kind) for e in range(experts) for layer, kind in expected}
        record = dict(experts=experts, shared_experts=0, parameters_per_expert=math.prod(dims),
                      expert_index=int(parts[5]), representation='per-expert')
    group = groups.setdefault(prefix, dict(seen=set(), required=required))
    if key in group['seen']:
        raise ValueError('duplicate CNN expert slice: ' + name)
    group['seen'].add(key)
    return record


def _check_cnn_banks(shape, groups):
    if shape.get('architecture') not in CNN_ARCHITECTURES:
        return
    expected = {f'{side}.{level}.{index}.ffn'
                for side, config in (('enc', 'blocks_enc'), ('dec', 'blocks_dec'))
                for level, count in shape[config].items() for index in range(count)}
    if set(groups) != expected:
        raise ValueError('CNN expert blocks do not match shape')
    for name, group in groups.items():
        if group['seen'] != group['required']:
            raise ValueError('incomplete CNN expert bank: ' + name)


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
    if name.startswith(('adapter.', 'stem.', 'head.', 'head_norm.')):
        return 'input_output'
    return 'residual_merge_other'


def audit_manifest(manifest, shape, *, prototype_budget=False):
    """Classify every tensor and reject malformed or inconsistent accounting."""
    totals = dict(parameters=0, public_parameters=0, expert_bank_parameters=0,
                  shared_expert_parameters=0, routed_expert_parameters=0,
                  selected_routed_equivalent=0, buffers=0)
    public = {}
    transitions = []
    tensors = []
    names = set()
    cnn_groups = {}
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
            cnn = _cnn_bank(name, dims, shape, cnn_groups)
            layout = _bank_layout(name, shape)
            suffix = name.rsplit('.', 1)[-1]
            bank = (layout is not None and
                    ((len(dims) == 3 and suffix in BANK_NAMES) or
                     (name.split('.')[0] in TRANSITIONS and len(dims) == 2 and suffix == 'bias')))
            if cnn is not None:
                totals['expert_bank_parameters'] += count
                totals['routed_expert_parameters'] += count
                # An individual expert bias need not divide evenly by E.
                # Sum exact rationals until the complete banks are checked.
                totals['selected_routed_equivalent'] += Fraction(count, cnn['experts'])
                record.update(category='expert_bank', **cnn)
            elif bank:
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

    _check_cnn_banks(shape, cnn_groups)
    selected = Fraction(totals['selected_routed_equivalent'])
    if selected.denominator != 1:
        raise ValueError('incomplete expert parameter accounting')
    totals['selected_routed_equivalent'] = int(selected)
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
    if prototype_budget and not new:
        raise ValueError('80M prototype budget applies only to shared/routed shapes')
    budget = dict(applicable=prototype_budget,
                  target_parameters=80_000_000 if prototype_budget else None, tolerance_fraction=0.01,
                  maximum_public_fraction=0.01 if prototype_budget else None, actual_public_fraction=fraction)
    if new:
        from moe_shape import ledger
        expected_layout = {e['name']: (e['shape'], e['isParameter']) for e in ledger(shape)}
        actual_layout = {e['name']: (e['shape'], e.get('isParameter', True)) for e in manifest['tensors']}
        expected = ({'enc_trans.' + n + '.' + s for n in ('d0', 'd1', 'd2', 'd3', 'd4') for s in ('weight', 'bias')} |
                    {'dec_trans.' + n + '.' + s for n in ('d0', 'd1', 'd2', 'd3') for s in ('weight', 'bias')} |
                    {n + '.' + s for n in ('vit_in', 'vit_out') for s in ('weight', 'bias')})
        budget.update(layout_pass=expected_layout == actual_layout,
                      transitions_pass=({t['name'] for t in transitions} == expected and
                                        all(t['category'] == 'expert_bank' for t in transitions)))
        if prototype_budget:
            budget.update(total_pass=79_200_000 <= totals['parameters'] <= 80_800_000,
                          public_pass=fraction <= 0.01)
        budget['pass'] = all(budget[key] for key in ('layout_pass', 'transitions_pass', 'total_pass', 'public_pass')
                             if key in budget)
    return dict(totals=totals, public_by_category=public, transitions=transitions,
                budget=budget, tensors=tensors,
                interpretation='Parameter equivalent across layers; not whole-image weight usage, MACs, or latency.')


def audit_package(directory, verify=False, *, prototype_budget=False):
    root = Path(directory).resolve()
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('model', {}).get('kind') != 'student':
        raise ValueError('expected a student package')
    shape_path = _relative(root, manifest['model'].get('shapeFile', 'shape.json'))
    shape = json.loads(shape_path.read_text(encoding='utf-8'))
    report = audit_manifest(manifest, shape, prototype_budget=prototype_budget)
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
    parser.add_argument('--prototype-budget', action='store_true', help='Also enforce the historical 80M / 1%% public budget')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit_package(args.weights, args.verify, prototype_budget=args.prototype_budget)
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
