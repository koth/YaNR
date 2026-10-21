"""从已有 manifest 导出 ROI jsonl,便于用 --roi-jsonl 重建清单而无需重跑人脸检测。只读源文件。"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument('manifest')
ap.add_argument('-o', '--out', required=True)
a = ap.parse_args()

with open(a.manifest, encoding='utf-8') as handle:
    manifest = json.load(handle)

rows = [entry for entry in manifest['entries'] if entry.get('rois')]
with open(a.out, 'w', encoding='utf-8') as handle:
    for entry in rows:
        handle.write(json.dumps({'path': entry['path'], 'rois': entry['rois']},
                                ensure_ascii=False) + '\n')
print(f'导出 {len(rows)} 张带 ROI 的条目 -> {a.out}')
