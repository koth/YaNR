from comet_ml import API


KEY_FILE = '/data2/hezm/klss/code/teacher/.comet_key'
EXPERIMENT = '13733cdbe561496e9ee12b3b908f7cb1'


def main():
    with open(KEY_FILE) as fh:
        key = fh.read().strip()
    api = API(api_key=key)
    print('default_workspace:', api.get_default_workspace())
    print('workspaces:', api.get_workspaces())
    exps = api.get_experiments(workspace='koth-chen', project_name='dlss-cnn-unet')
    print('project experiments:', len(exps))
    for exp in exps[:10]:
        print(' ', exp.id, exp.name, exp.url)
    exp = api.get_experiment_by_key(EXPERIMENT)
    if exp is None:
        print('target experiment: NOT FOUND')
        return
    print('target:', exp.id, exp.name, exp.url)
    metrics = exp.get_metrics()
    print('metric rows:', 0 if metrics is None else len(metrics))
    if metrics:
        from collections import Counter
        names = Counter(row.get('metricName') for row in metrics)
        print('metric names:', names.most_common(20))
        for wanted in ('loss/out', 'loss/feature', 'psnr/teacher', 'train/ms'):
            rows = [r for r in metrics if r.get('metricName') == wanted]
            if rows:
                print(wanted, 'last:', rows[-1].get('metricValue'), 'step', rows[-1].get('step'))
        for row in metrics[-8:]:
            print(' ', row.get('metricName'), row.get('metricValue'), 'step', row.get('step'))


if __name__ == '__main__':
    main()
