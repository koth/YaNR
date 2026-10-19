"""训练损失的逐层分解(comet 拉取)与 vis 上的像素权重统计,用来判断梯度质量落在哪。

用法:
  python3 loss_breakdown.py series <ws> <proj> <exp> <keyfile>       # 关键指标时间序列
  python3 loss_breakdown.py weight <vis-dir> [...]                   # 该 vis 样本的 w 统计
"""
import sys

ACT_FLOOR, ACT_KNEE = 0.1, 0.02


def series(argv):
    from comet_ml import API
    api = API(api_key=open(argv[4]).read().strip())
    exp = api.get_experiment(argv[1], argv[2], argv[3])
    rows = exp.get_metrics()
    want = ['feat/s-dec-d0', 'feat/s-enc-d0', 'feat/s-vit', 'feat/s-enc-full',
            'loss/feature', 'loss/out', 'loss/texture', 'loss/detail',
            'psnr/teacher', 'moe/dead_frac', 'moe/entropy', 'moe/util_max', 'moe/z']
    by = {}
    for m in rows:
        name, value, step = m.get('metricName'), m.get('metricValue'), m.get('step') or 0
        if name in want and value is not None:
            by.setdefault(name, {})[step] = float(value)
    steps = sorted(set().union(*[set(v) for v in by.values()])) if by else []
    keep = [s for s in steps if s % 2000 == 0]
    print('step    ' + '  '.join(f'{n.split("/")[-1]:>10}' for n in want))
    for s in keep:
        print(f'{s:7d} ' + '  '.join(
            f'{by.get(n, {}).get(s, float("nan")):10.4f}' for n in want))


def weight(argv):
    import numpy as np
    from PIL import Image
    for d in argv[1:]:
        proxy = np.asarray(Image.open(f'{d}/proxy.png').convert('RGB'), dtype=np.float64) / 255
        teacher = np.asarray(Image.open(f'{d}/teacher.png').convert('RGB'), dtype=np.float64) / 255
        student = np.asarray(Image.open(f'{d}/student.png').convert('RGB'), dtype=np.float64) / 255
        delta = np.abs(teacher - proxy).mean(2)
        w = ACT_FLOOR + (1 - ACT_FLOOR) * delta ** 2 / (delta ** 2 + ACT_KNEE ** 2)
        err = np.abs(teacher - student).mean(2)
        sat = ((teacher <= 0.0005) | (teacher >= 0.9995)).mean()
        print(f'{d}: mean(w)={w.mean():.3f}  加权图像 L1={float((err * w).mean()):.5f}  '
              f'未加权={err.mean():.5f}  teacher 饱和像素占比={sat * 100:.1f}%  '
              f'delta p50={np.median(delta) * 255:.2f}/255 p95={np.percentile(delta, 95) * 255:.2f}/255')


if __name__ == '__main__':
    if sys.argv[1] == 'series':
        series(sys.argv[1:])
    else:
        weight(sys.argv[1:])
