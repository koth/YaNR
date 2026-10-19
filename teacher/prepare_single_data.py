#!/usr/bin/env python3
"""Index unaugmented images, group-isolated holdout and optional face ROIs."""
import argparse
import json
from pathlib import Path

from single_frame_data import build_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--out', help='defaults to config data_manifest')
    parser.add_argument('--roi-jsonl', help='lines: {path, rois: [[x0,y0,x1,y1], ...]}')
    parser.add_argument('--detect-faces', action='store_true', help='offline OpenCV Haar face boxes')
    parser.add_argument('--groups-jsonl', help='lines: {path, group}; same video/person group across sources')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as handle:
        cfg = json.load(handle)
    manifest = build_manifest(cfg, args.data_root, args.roi_jsonl, args.detect_faces, args.groups_jsonl)
    if cfg['sampling'].get('roi_native', 0) and not any(
            x['split'] == 'train' and x['rois'] and
            min(x['width'], x['height']) >= min(cfg['size']) for x in manifest['entries']):
        raise SystemExit('no eligible train ROI; use --detect-faces or --roi-jsonl; manifest was not written')
    target = Path(args.out or cfg['data_manifest'])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(manifest['counts'], ensure_ascii=False))
    print(json.dumps(manifest['source_counts'], ensure_ascii=False))
    print('manifest:', target)


if __name__ == '__main__':
    main()
