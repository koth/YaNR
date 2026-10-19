"""检查 routing.jsonl 里 shared-routed MoE 的路由健康度(只读)。

用法: python3 routing_health.py <routing.jsonl> [最近 N 条,默认 5]
输出每条记录的:专家负载份额 max/mean、死专家数、归一化熵、bias 极差、gate/分数 margin,
以及逐层最差/最好的负载分布,用来尽早发现路由塌缩(旧 v0 MoE 的病)。
"""
import json
import math
import sys

import numpy as np


def layer_stats(lay):
    load = np.asarray(lay['load_rows'], dtype=np.float64)
    total = load.sum()
    share = load / total if total else load
    p = share[share > 0]
    ent = float(-(p * np.log(p)).sum() / math.log(len(load))) if len(p) and len(load) > 1 else 1.0
    bias = np.asarray(lay['bias'], dtype=np.float64)
    return dict(max_share=float(share.max()), dead=int((load == 0).sum()), entropy=ent,
                bias_span=float(bias.max() - bias.min()),
                gate_mean=lay.get('gate_mean'), margin=lay.get('score_margin_mean'))


def main():
    path = sys.argv[1]
    tail = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    rows = [json.loads(line) for line in open(path)]
    print(f'{len(rows)} records, step {rows[0]["step"]}..{rows[-1]["step"]}')
    for rec in rows[-tail:]:
        stats = [layer_stats(l) for l in rec['layers'].values()]
        ms = np.array([s['max_share'] for s in stats])
        dead = sum(s['dead'] for s in stats)
        ent = np.mean([s['entropy'] for s in stats])
        span = np.mean([s['bias_span'] for s in stats])
        margin = np.mean([s['margin'] or 0 for s in stats])
        print(f"step {rec['step']:>6}  layers={len(stats)}  max_share: mean={ms.mean():.3f} "
              f"worst={ms.max():.3f}  dead_experts={dead}  entropy={ent:.3f}  "
              f"bias_span={span:.4f}  margin={margin:.3f}")
        worst = sorted(zip(ms, rec['layers']), key=lambda t: -t[0])[:2]
        print('        worst: ' + ', '.join(f'{n} {v:.3f}' for v, n in worst))


if __name__ == '__main__':
    main()
