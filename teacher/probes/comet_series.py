"""按步打印 comet 上若干指标的时间序列(用于判断"是一直如此还是近期退化")。

    python3 comet_series.py <workspace> <project> <experiment_key> <key_file> <metric> [...]

每个指标打 step 与值的配对,按 step 升序。
"""
import sys

from comet_ml import API


def main():
    workspace, project, key, key_file = sys.argv[1:5]
    wanted = sys.argv[5:]
    api = API(api_key=open(key_file).read().strip())
    exp = api.get_experiment(workspace, project, key)
    series = {name: [] for name in wanted}
    for m in exp.get_metrics():
        name = m.get('metricName')
        if name in series and m.get('metricValue') is not None:
            series[name].append((m.get('step') or 0, m['metricValue']))
    for name in wanted:
        rows = [(float(s), float(v)) for s, v in series[name] if s is not None]
        rows.sort()
        if not rows:
            print(f'{name}: 无数据')
            continue
        picked = rows[::max(1, len(rows) // 14)]
        print(f'{name}  (n={len(rows)})')
        print('   ' + '  '.join(f'{int(s) if s == int(s) else s}:{v:.4f}'
                                for s, v in picked))


if __name__ == '__main__':
    main()
