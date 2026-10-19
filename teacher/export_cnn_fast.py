#!/usr/bin/env python3
"""Export a fast CNN checkpoint to PyTorch packed or existing native CNN DSL."""
import argparse
import json
from pathlib import Path

import torch

from nr_cnn_unet_fast import CNNUNetFast, load_fast_shape
from nr_geometry import geometry_from_valid
from nr_package import build_student, read_package, write_package
from single_frame_data import digest_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, nargs='?')
    parser.add_argument('--shape', type=Path)
    parser.add_argument('--legacy-package', type=Path, help='pack existing CNN weights without compression/retraining')
    parser.add_argument('--state', choices=('ema', 'model'), default='ema')
    parser.add_argument('--backend', choices=('native', 'pytorch'), default='native')
    parser.add_argument('-o', '--out', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.checkpoint) == bool(args.legacy_package):
        parser.error('specify one checkpoint or --legacy-package')
    if (args.out / 'manifest.json').exists():
        parser.error('output package already exists; choose a new directory')
    geometry = geometry_from_valid(320, 320)
    if args.legacy_package:
        legacy = build_student(args.legacy_package / 'shape.json', geometry)
        if getattr(legacy, 'cfg', {}).get('architecture') != 'cnn_unet_moe8':
            parser.error('legacy package must be cnn_unet_moe8')
        state, metadata = read_package(args.legacy_package, verify=True)
        legacy.load_state_dict(state)
        model = CNNUNetFast.from_legacy(legacy)
        step = metadata['model'].get('step', -1)
        extra = dict(weightSource='legacy-package', sourceVersion=metadata['model']['version'],
                     sourceManifestSha256=digest_file(args.legacy_package / 'manifest.json'),
                     version=metadata['model']['version'] + '-grouped')
        del state, legacy
    else:
        if args.shape is None:
            parser.error('--shape is required with a checkpoint')
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if args.state not in checkpoint:
            parser.error('checkpoint has no ' + args.state + ' state')
        model = CNNUNetFast(load_fast_shape(args.shape), geometry)
        model.load_state_dict(checkpoint[args.state])
        step = checkpoint.get('step', -1)
        extra = dict(weightSource=args.state, checkpointSha256=digest_file(args.checkpoint))
    model.eval().switch_to_deploy()
    if args.backend == 'native':
        model = model.to_native_model()
    args.out.mkdir(parents=True, exist_ok=True)
    shape_file = args.out / 'shape.json'
    shape_file.write_text(json.dumps(model.cfg, indent=2), encoding='utf-8')
    extra.update(sourceArchitecture='cnn-unet-fast8-v1', repBranchesFused=True)
    manifest = write_package(model, model.state_dict(), args.out, shape_file, step=step, extra=extra)
    print(manifest['model']['version'], manifest['model']['params'], 'parameters ->', args.out)


if __name__ == '__main__':
    main()
