#!/usr/bin/env python3
"""Generate and smoke-test an UNTRAINED 80M shared/routed speed prototype.

python teacher/prototype_shared_moe.py --output tmp/shared_moe/prototype
All outputs stay separate from trained weights and sample images.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audit_moe import audit_manifest, audit_package, sha256
from moe_cost import account_moe
from moe_shape import candidate, ledger
from nr_geometry import geometry_from_valid
from nr_package import load_student_package, write_package
from nr_student import StudentNetwork


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('tmp/shared_moe/prototype'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--preset', choices=('balanced', 'compact'), default='balanced')
    parser.add_argument('--teacher-weights', type=Path, default=Path('weights/teacher'))
    parser.add_argument('--sizes', default='320x320,512x512,320x512,137x201')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    # Same 17-expert structure, two legal widths to keep the budget choice visible.
    candidates = [candidate([32, 32, 64, 96, 192, 192], [32, 32, 64, 64, 64, 64], 384, h)
                  for h in (736, 768)]
    compact = candidate([32, 32, 32, 64, 128, 192], [32, 32, 32, 32, 32, 32], 384, 928)
    candidates.append(compact)
    selected = compact if args.preset == 'compact' else candidates[1]
    selected['comment'] = 'UNTRAINED random FP32 prototype for correctness/speed only; no image quality claims.'
    shape_path = args.output / 'shape.json'
    shape_path.write_text(json.dumps(selected, indent=2) + '\n', encoding='utf-8')
    geometry = geometry_from_valid(320, 320)
    model = StudentNetwork(selected, geometry).to(args.device).eval()
    expected = {e['name']: tuple(e['shape']) for e in ledger(selected)}
    actual = {n: tuple(t.shape) for n, t in model.state_dict().items()}
    if expected != actual:
        raise AssertionError(dict(missing=expected.keys() - actual.keys(), extra=actual.keys() - expected.keys(),
                                  mismatches=[n for n in actual if n in expected and actual[n] != expected[n]]))
    from moe_training import provenance
    write_package(model, model.state_dict(), args.output, shape_path,
                  extra=dict(untrainedPrototype=True, initializationSeed=args.seed, biasSource='zero-initialized',
                             source=provenance(shape_path, args.teacher_weights), generator='teacher/prototype_shared_moe.py'))
    audit = audit_package(args.output, verify=True, prototype_budget=True)
    if not audit['budget']['pass']:
        raise AssertionError(audit['budget'])
    loaded, _ = load_student_package(args.output, 320, 320, args.device)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name]), name
    results = []
    with torch.inference_mode():
        for index, size in enumerate(args.sizes.split(',')):
            width, height = map(int, size.split('x'))
            g = geometry_from_valid(width, height)
            model.g = loaded.g = g
            rng = np.random.default_rng([args.seed, index])
            data = rng.normal(0, 0.2, (g['full_rows'], 16)).astype(np.float32)
            features = torch.from_numpy(data).to(args.device)
            for module in model.modules():
                if hasattr(module, 'router_bias'):
                    module.capture_enabled = True
            output = model(features)
            restored = loaded(features)
            repeated = model(features)
            assert output.shape == (g['full_rows'], 4) and torch.isfinite(output).all()
            assert torch.equal(output, restored) and torch.equal(output, repeated), size
            branches = {}
            for name, module in model.named_modules():
                if hasattr(module, 'router_bias'):
                    captures = module.captures
                    branches[name] = dict(rows=int(captures['assign'].numel()),
                                          load=torch.bincount(captures['assign'], minlength=16).tolist(),
                                          gate_mean=float(captures['gate'].mean()),
                                          gate_max=float(captures['gate'].max()))
                    if 'ffn_shared' in captures:
                        branches[name].update(shared_rms=float(captures['ffn_shared'].square().mean().sqrt()),
                                              routed_rms=float(captures['ffn_routed'].square().mean().sqrt()))
            load = {n: b['load'] for n, b in branches.items()}
            results.append(dict(valid=[width, height], field=[g['full_width'], g['full_height']],
                                finite=True, package_bit_exact=True, repeat_bit_exact=True,
                                output_abs_max=float(output.abs().max()), branches=branches,
                                cost=account_moe(selected, g, load)))
            print(f'{size}: field {g["full_width"]}x{g["full_height"]}, finite, package/repeat bit-exact', flush=True)
    report = dict(seed=args.seed, device=args.device, shape_sha256=sha256(shape_path), audit=audit,
                  candidates=[dict(shape=c, cost_512=account_moe(c, geometry_from_valid(512, 512)),
                                   **audit_manifest({'tensors': ledger(c)}, c)['totals']) for c in candidates],
                  results=results)
    path = args.output / 'reference.json'
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Saved', path)


if __name__ == '__main__':
    main()
