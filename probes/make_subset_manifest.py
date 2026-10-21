"""实验 B 第 1 步:从正式清单裁一个小子集(默认 64 训练 + 8 验证)。

保持原有 source 分布:按 source 的 entries 顺序等间隔抽取,避免只拿到某一个源。
    python3 probes/make_subset_manifest.py --train 64 --val 8
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

SOURCE = Path('/f/yanr/keep_tmp/single_v1/manifest.json')
OUT = Path('/f/yanr/keep_tmp/overfit_v1')


def pick(entries, count):
    """按 source 轮转抽取,保留分布;数量不足时允许重复。"""
    by_source = defaultdict(list)
    for entry in entries:
        by_source[entry['source']].append(entry)
    names = sorted(by_source)
    chosen, cursor = [], {name: 0 for name in names}
    while len(chosen) < count:
        for name in names:
            if len(chosen) >= count:
                break
            items = by_source[name]
            if not items:
                continue
            chosen.append(items[cursor[name] % len(items)])
            cursor[name] += 1
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=int, default=64)
    ap.add_argument('--val', type=int, default=8)
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()

    manifest = json.loads(SOURCE.read_text(encoding='utf-8'))
    train = pick([e for e in manifest['entries'] if e['split'] == 'train'], a.train)
    val_pool = [e for e in manifest['entries'] if e['split'] == 'val']
    val = pick(val_pool, a.val)

    # 子集里把抽出来的训练条目整体标成 train、验证条目标成 val;分组保持不变。
    sub = dict(version=1, seed=manifest['seed'], sources=manifest['sources'],
               entries=[dict(e) for e in train] + [dict(e) for e in val],
               rejected=[], counts={}, source_counts={})
    counts = defaultdict(lambda: dict(retained=0, groups=0, train=0, val=0, roi_images=0))
    groups = defaultdict(set)
    for entry in sub['entries']:
        record = counts[entry['source']]
        record['retained'] += 1
        record[entry['split']] += 1
        record['roi_images'] += bool(entry['rois'])
        groups[entry['source']].add(entry['split_group'])
    for name, record in counts.items():
        record['groups'] = len(groups[name])
    sub['source_counts'] = dict(counts)
    sub['counts'] = dict(scanned=len(sub['entries']), retained=len(sub['entries']),
                         duplicates=0, rejected=0,
                         train=sum(e['split'] == 'train' for e in sub['entries']),
                         val=sum(e['split'] == 'val' for e in sub['entries']))
    a.out.mkdir(parents=True, exist_ok=True)
    target = a.out / 'manifest.json'
    target.write_text(json.dumps(sub, ensure_ascii=False, indent=1), encoding='utf-8')
    print('写出', target)
    print('train', sub['counts']['train'], 'val', sub['counts']['val'])
    print({k: v['retained'] for k, v in sub['source_counts'].items()})
    # 训练用路径清单,方便第 3 步在同一批图上评估
    paths = [e['path'] for e in sub['entries'] if e['split'] == 'train']
    (a.out / 'train_paths.json').write_text(json.dumps(paths, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
