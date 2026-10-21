"""就地修正单组源的 train/val 切分,不重跑整份清单。

背景:按源切分时,只有一个组的源(gameplay / frames_2k 用 group_by=directory)
无法切分,却被哈希分数直接判进 val,导致训练集里一条都不剩。
genuine 的修复是 build_manifest 里的 `if len(owned) < 2: continue`,但对已经
建好的清单来说,唯一正确的差异就是"单组源全部改回 train" —— 多组源在两个版本
下的判据完全一致,所以不需要重建。

用法:
    python3 fix_single_group_split.py manifest.json [-o manifest.fixed.json]
"""
import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('manifest')
    ap.add_argument('-o', '--out')
    a = ap.parse_args()

    with open(a.manifest, encoding='utf-8') as handle:
        manifest = json.load(handle)
    entries = manifest['entries']

    groups_of = defaultdict(set)
    for entry in entries:
        groups_of[entry['source']].add(entry['split_group'])
    single = {name for name, groups in groups_of.items() if len(groups) < 2}
    if not single:
        print('没有单组源,无需修改')
        return 0

    moved = 0
    for entry in entries:
        if entry['source'] in single and entry['split'] != 'train':
            entry['split'] = 'train'
            moved += 1

    for name, record in manifest['source_counts'].items():
        items = [x for x in entries if x['source'] == name]
        record['train'] = sum(x['split'] == 'train' for x in items)
        record['val'] = sum(x['split'] == 'val' for x in items)
    manifest['counts']['train'] = sum(x['split'] == 'train' for x in entries)
    manifest['counts']['val'] = sum(x['split'] == 'val' for x in entries)

    target = a.out or a.manifest
    with open(target, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=1)

    print(f'单组源: {sorted(single)}')
    print(f'改回 train 的条目: {moved}')
    for name in sorted(single):
        record = manifest['source_counts'][name]
        print(f'  {name}: train={record["train"]} val={record["val"]}')
    print(f'总计 train={manifest["counts"]["train"]} val={manifest["counts"]["val"]}')
    print(f'写出 {target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
