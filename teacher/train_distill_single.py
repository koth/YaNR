#!/usr/bin/env python3
"""Single-frame online distillation for the independent 56M / 7M CNN.

No pair, motion, temporal blend or synthetic degradation path. Teacher and
student consume the SAME cropped/resized feature array and first-frame history.
Supports rectangular valid regions, bounded microbatches and fixed holdout eval.
"""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

import numpy as np
import torch

from nr_package import build_student
from nr_precision import tf32_matmul
from single_frame_data import SingleFrameSource, digest_file


ROOT = Path(__file__).resolve().parent.parent
TEACHER_DIMS = {
    's-enc-full': 32, 's-enc-d0': 32, 's-enc-d1': 64, 's-enc-d2': 128,
    's-enc-d3': 256, 's-enc-d4': 512, 's-vit': 1024,
    's-dec-d4': 512, 's-dec-d3': 256, 's-dec-d2': 128,
    's-dec-d1': 64, 's-dec-d0': 32,
}


def load_config(path):
    with open(path, encoding='utf-8') as handle:
        cfg = json.load(handle)
    for key in ('pair_fraction', 'loss_temporal', 'loss_logit', 'degrade_prob'):
        if float(cfg.get(key, 0)) != 0:
            raise ValueError(key + ' must be zero for this single-frame entry point')
    for key in ('batch', 'steps', 'batch_rows', 'teacher_batch_rows', 'prep_workers', 'log_every', 'ema_every'):
        if int(cfg[key]) < 1:
            raise ValueError(key + ' must be positive')
    if int(cfg['val_count']) < 1 or int(cfg['validation_max_rows']) < 1:
        raise ValueError('validation count and row budget must be positive')
    if cfg.get('validation_weight_source', 'ema') not in ('ema', 'raw'):
        raise ValueError('validation_weight_source must be ema or raw')
    for key in ('shape', 'data_manifest'):
        if not Path(cfg[key]).is_absolute():
            cfg[key] = str(ROOT / cfg[key])
    for source in cfg['sources']:
        if source['weight'] < 0 or int(source.get('frame_stride', 1)) < 1:
            raise ValueError('source weights/strides must be nonnegative/positive')
    return cfg


def load_helpers():
    # Defer teacher imports so --help and manifest tools work without CUDA.
    import train_distill as losses
    from run_image import build_features
    from nr_student import student_alignment
    return losses, build_features, student_alignment()


def configure_teacher_library():
    """Honor the A100 architecture without changing the old teacher loader."""
    library = os.environ.get('NR_CUDA_LIBRARY')
    if library:
        path = Path(library).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
    elif os.name == 'nt':
        path = ROOT / 'tmp/shared_moe/nr_kernels.dll'
        if not path.is_file():
            raise RuntimeError('Windows teacher requires a prebuilt DLL; set NR_CUDA_LIBRARY')
    else:
        arch = os.environ.get('NR_CUDA_ARCH', 'sm_%d%d' % torch.cuda.get_device_capability())
        if not re.fullmatch(r'sm_[0-9]+', arch):
            raise ValueError('NR_CUDA_ARCH must be sm_XX')
        source = ROOT / 'teacher/nr_kernels.cu'
        cache = ROOT / 'tmp/single_frame_kernels'
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / ('nr_kernels_' + arch + '_' + digest_file(source)[:16] + '.so')
        if not path.is_file():
            temporary = path.with_suffix('.%d.tmp.so' % os.getpid())
            command = [os.environ.get('NVCC', '/usr/local/cuda/bin/nvcc'), '-O3', '-std=c++14',
                       '-arch=' + arch, '-cudart=static', '-shared', '-Xcompiler', '-fPIC',
                       str(source), '-o', str(temporary)]
            print('building teacher:', ' '.join(command), flush=True)
            subprocess.run(command, check=True)
            temporary.replace(path)
    os.environ['NR_CUDA_LIBRARY'] = str(path)
    import nr_cuda
    nr_cuda._SO = str(path)
    print('teacher native library:', path, flush=True)


def make_aligner(model, losses):
    dims = {'s-vit': model.channels['d5']}
    dims.update({'s-enc-' + k: model.channels[k] for k in model.channels if k != 'd5'})
    dims.update({'s-dec-' + k: model.channels[k] for k in model.channels if k != 'd5'})
    return losses.FeatureAligner({k.replace('-', '_'): (dims[k], t) for k, t in TEACHER_DIMS.items()})


class TeacherService:
    """One teacher weight copy; its geometry is changed only inside this lock."""
    def __init__(self, weights, feature_tags, budget, device):
        if os.environ.get('NR_CUDA_LIBRARY'):
            import nr_cuda
            nr_cuda._SO = str(Path(os.environ['NR_CUDA_LIBRARY']).resolve())
        from annotate import TeacherAnnotator
        from nr_geometry import geometry_from_valid
        self.annotator = TeacherAnnotator(weights, geometry_from_valid(320, 320))
        self.tags, self.budget = tuple(feature_tags), int(budget)
        self.lock = threading.Lock()
        self.stream = torch.cuda.Stream(device=device)

    def prepare(self, items, capture=True):
        grouped = defaultdict(list)
        for item in items:
            grouped[item['hw']].append(item)
        with self.lock, torch.cuda.stream(self.stream), torch.inference_mode():
            for (h, w), group in grouped.items():
                g = group[0]['geometry']
                if g['full_rows'] > self.budget:
                    raise ValueError('teacher row budget too small for one field: ' + str((w, h)))
                ann = self.annotator
                ann.g = ann.network.g = g
                limit = max(1, self.budget // g['full_rows'])
                for start in range(0, len(group), limit):
                    chunk = group[start:start + limit]
                    b = len(chunk)
                    features = np.stack([p['features'] for p in chunk]).reshape(-1, 16)
                    tags = self.tags if capture else ()
                    res = ann.annotate(features, capture=tags, batch=b)
                    heads = res['head'].reshape(b, g['full_height'], g['full_width'], 4)
                    for index, item in enumerate(chunk):
                        item['t_head'] = heads[index, :h, :w].copy()
                        item['t_res'] = {name: np.split(res[name], b, axis=0)[index] for name in tags}
        return items


def attach_features(item, build_features):
    c = item['controls']
    item['features'] = build_features(item['proxy'], item['geometry'], item['noise_seed'],
                                      c['style'], c['tone'], c['structure'], c['skin'],
                                      c['auto_mask'], history=None)
    return item


def chunk_loss(model, aligner, chunk, cfg, helpers):
    """Also used by CPU semantic tests with deterministic synthetic teacher labels."""
    losses, _, alignment = helpers
    device = next(model.parameters()).device
    b, (h, w) = len(chunk), chunk[0]['hw']
    g = chunk[0]['geometry']
    if any(item['hw'] != (h, w) for item in chunk):
        raise ValueError('microbatch must have one valid geometry')
    model.g = g
    features = torch.from_numpy(np.stack([p['features'] for p in chunk])).reshape(-1, 16).to(device)
    with tf32_matmul(cfg['student_tf32']):
        head = model(features, batch=b)
    head = head.reshape(b, g['full_height'], g['full_width'], 4)[:, :h, :w].float()
    proxy = torch.from_numpy(np.stack([p['proxy'] for p in chunk])).to(device)
    target = torch.from_numpy(np.stack([p['t_head'] for p in chunk])).to(device)
    out, predicted, teacher, _, _, _ = losses.output_losses(
        head, target, proxy, repro=None, logit_w=0.,
        act_floor=cfg['action_floor'], act_knee=cfg['action_knee'],
        tex_w=cfg['loss_texture'], tex_floor=cfg['texture_floor'],
        tex_knee=cfg['texture_knee'], tex_boost=0., tex_sigma=cfg['tex_sigma'],
        texstat_w=cfg['loss_texstat'], texstat_boost=0.)
    detail = losses.laplacian_l1(predicted, teacher) + losses.sobel_l1(predicted, teacher)
    feature = head.new_zeros(())
    if cfg['loss_feature']:
        captures = {name: np.concatenate([p['t_res'][name] for p in chunk], axis=0)
                    for name in alignment.values() if name != 'head'}
        feature, _ = losses.feature_losses(model.captures, captures, aligner, {}, diagnostics=False)
    aux, z = model.moe_losses()
    image_mse = (predicted - teacher).square().mean()
    loss = (cfg['loss_out'] * out.mean() + cfg['loss_detail'] * detail.mean()
            + cfg['loss_feature'] * feature + cfg.get('loss_mse', 0.) * image_mse
            + cfg['moe_aux'] * aux + cfg['moe_z'] * z)
    mse = (predicted.detach() - teacher.detach()).square().reshape(b, -1).mean(1)
    psnr = (-10 * mse.clamp_min(1e-12).log10()).mean()
    metrics = dict(loss=loss.detach(), out=out.detach().mean(), mse=image_mse.detach(), detail=detail.detach().mean(),
                   feature=feature.detach(), moe_aux=aux.detach(), moe_z=z.detach(),
                   psnr_teacher=psnr)
    return loss, metrics


def provenance(cfg):
    names = ('nr_cnn_single.py', 'nr_cnn_unet_fast.py', 'nr_cnn_unet.py', 'nr_shared_moe.py',
             'nr_package.py', 'single_frame_data.py', 'train_distill_single.py',
             'train_distill.py', 'nr_geometry.py', 'run_image.py', 'annotate.py',
             'nr_model.py', 'nr_network.py', 'nr_torch.py', 'nr_cuda.py', 'nr_kernels.cu',
             'nr_numerics.py', 'nr_precision.py')
    stamp = dict(shape=digest_file(cfg['shape']), data_manifest=digest_file(cfg['data_manifest']),
                 teacher_manifest=digest_file(Path(cfg['teacher_weights']) / 'manifest.json'),
                 sources={name: digest_file(ROOT / 'teacher' / name) for name in names})
    if os.environ.get('NR_CUDA_LIBRARY'):
        stamp['teacher_native_library'] = digest_file(os.environ['NR_CUDA_LIBRARY'])
    return stamp


def validate(model, ema, source, teacher, cfg, build_features, step, out):
    """Fixed held-out center crops AND native images, with explicit oversize skips."""
    was_training, capture, geometry = model.training, model.capture_enabled, model.g
    saved = None
    if cfg.get('validation_weight_source', 'ema') == 'ema':
        saved = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema)
    model.eval()
    model.capture_enabled = False
    records = []
    try:
        with torch.inference_mode():
            for mode in cfg.get('validation_modes', ['crop', 'native']):
                for index in range(min(int(cfg['val_count']), len(source.val))):
                    item = source.validation(index, mode)
                    g = item['geometry']
                    if g['full_rows'] > cfg['validation_max_rows']:
                        records.append(dict(mode=mode, path=item['metadata']['path'],
                                            skipped='field exceeds validation_max_rows',
                                            full_rows=g['full_rows']))
                        continue
                    attach_features(item, build_features)
                    # Validation has a separate per-image budget, no resize fallback.
                    old_budget = teacher.budget
                    try:
                        teacher.budget = cfg['validation_max_rows']
                        teacher.prepare([item], capture=False)
                    finally:
                        teacher.budget = old_budget
                    model.g = g
                    device = next(model.parameters()).device
                    x = torch.from_numpy(item['features']).to(device)
                    with tf32_matmul(cfg['student_tf32']):
                        head = model(x).reshape(g['full_height'], g['full_width'], 4)
                    h, w = item['hw']
                    residual = head[:h, :w, :3].float().cpu().numpy() / 4
                    predicted = np.clip(item['proxy'] + residual, 0, 1)
                    target = np.clip(item['proxy'] + item['t_head'][..., :3] / 4, 0, 1)
                    mse = float(np.square(predicted.astype(np.float64) - target).mean())
                    baseline = float(np.square(item['proxy'].astype(np.float64) - target).mean())
                    record = dict(item['metadata'], mode=mode, noise_seed=item['noise_seed'],
                                  mse=mse, psnr_teacher=-10 * np.log10(max(mse, 1e-12)),
                                  psnr_input=-10 * np.log10(max(baseline, 1e-12)))
                    mask = np.zeros((h, w), dtype=bool)
                    for x0, y0, x1, y1 in item['metadata']['valid_rois']:
                        mask[y0:y1, x0:x1] = True
                    if mask.any():
                        roi_mse = float(np.square(predicted[mask].astype(np.float64) - target[mask]).mean())
                        roi_base = float(np.square(item['proxy'][mask].astype(np.float64) - target[mask]).mean())
                        record.update(roi_pixels=int(mask.sum()),
                                      psnr_roi=-10 * np.log10(max(roi_mse, 1e-12)),
                                      psnr_roi_input=-10 * np.log10(max(roi_base, 1e-12)))
                    records.append(record)
    finally:
        if saved is not None:
            model.load_state_dict(saved)
        model.g, model.capture_enabled = geometry, capture
        model.train(was_training)
    means = {}
    for mode in cfg.get('validation_modes', ['crop', 'native']):
        subset = [r for r in records if r['mode'] == mode and 'mse' in r]
        if subset:
            means[mode] = dict(psnr_teacher=float(np.mean([r['psnr_teacher'] for r in subset])),
                               psnr_input=float(np.mean([r['psnr_input'] for r in subset])), count=len(subset))
        else:
            means[mode] = dict(count=0)
        roi_subset = [r for r in subset if 'psnr_roi' in r]
        if roi_subset:
            means[mode].update(psnr_roi=float(np.mean([r['psnr_roi'] for r in roi_subset])),
                               psnr_roi_input=float(np.mean([r['psnr_roi_input'] for r in roi_subset])),
                               roi_count=len(roi_subset))
    report = dict(step=step, weight_source=cfg.get('validation_weight_source', 'ema'),
                  means=means, images=records)
    (out / ('validation_%06d.json' % step)).write_text(json.dumps(report, indent=1), encoding='utf-8')
    print('validation:', json.dumps(means), flush=True)
    return means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', help='override prepared data manifest')
    parser.add_argument('--teacher-weights')
    parser.add_argument('-o', '--out', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--steps', type=int, help='override total steps, useful for a separate smoke run')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--val-count', type=int)
    parser.add_argument('--comet-disabled', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    for key, value in (('steps', args.steps), ('batch', args.batch), ('val_count', args.val_count)):
        if value is not None:
            if value < 1:
                parser.error(key + ' must be positive')
            cfg[key] = value
    if args.comet_disabled:
        cfg['comet_disabled'] = True
    if args.manifest:
        cfg['data_manifest'] = str(Path(args.manifest).resolve())
    if args.teacher_weights:
        cfg['teacher_weights'] = args.teacher_weights
    if not torch.cuda.is_available():
        raise SystemExit('training requires CUDA; CPU tests: python -m unittest discover -s teacher -p test_single_frame.py')
    configure_teacher_library()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False  # router/head Conv2d stay FP32 too
    device = torch.device('cuda')
    source = SingleFrameSource(cfg['data_manifest'], cfg)
    with open(cfg['shape'], encoding='utf-8') as shape_file:
        shape = json.load(shape_file)
    from nr_geometry import geometry_from_valid
    model = build_student(cfg['shape'], geometry_from_valid(320, 320), device)
    model.capture_enabled = bool(cfg['loss_feature'])
    helpers = load_helpers()
    losses, build_features, alignment = helpers
    aligner = make_aligner(model, losses).to(device)
    params = list(model.parameters()) + (list(aligner.parameters()) if cfg['loss_feature'] else [])
    optimizer = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['wd'], fused=True)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    audit = model.parameter_ledger()
    expected = shape.get('parameter_targets', {})
    for key, value in expected.items():
        if abs(audit[key] - value) > value * .01:
            raise ValueError('model does not meet parameter target ' + key)
    print('parameters:', json.dumps({k: v for k, v in audit.items() if k != 'blocks'}), flush=True)
    stamp = provenance(cfg)
    start = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        if ckpt['provenance'] != stamp:
            raise ValueError('resume teacher/model/data provenance mismatch')
        for key in ('seed', 'batch', 'size', 'size_weights', 'sampling', 'controls', 'sources',
                    'context_buckets', 'crop_scale'):
            if ckpt['config'][key] != cfg[key]:
                raise ValueError('resume sampling mismatch: ' + key)
        model.load_state_dict(ckpt['model'])
        aligner.load_state_dict(ckpt['aligner'])
        optimizer.load_state_dict(ckpt['optim'])
        ema = {k: v.to(device) for k, v in ckpt['ema'].items()}
        torch.set_rng_state(ckpt['torch_rng'])
        torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        start = int(ckpt['step']) + 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists() and not args.resume:
        raise ValueError('run directory already contains config.json; use a new directory or --resume')
    (out / 'config.json').write_text(json.dumps(cfg, indent=1), encoding='utf-8')
    (out / 'parameter_ledger.json').write_text(json.dumps(audit, indent=1), encoding='utf-8')
    (out / 'provenance.json').write_text(json.dumps(stamp, indent=1), encoding='utf-8')

    experiment = None
    if not cfg.get('comet_disabled', False):
        key = losses.comet_key(cfg)
        if key and losses.Experiment is not None:
            experiment = losses.Experiment(api_key=key, project_name=cfg['comet_project'], log_code=False)
            experiment.log_parameters({k: v for k, v in cfg.items() if k != 'comet_key'})
        else:
            print('Comet unavailable or no key; metrics are written locally.', flush=True)

    teacher = TeacherService(cfg['teacher_weights'],
                             [t for t in alignment.values() if t != 'head'] if cfg['loss_feature'] else [],
                             cfg['teacher_batch_rows'], device)
    def prepare_item(index):
        item = source.sample(index)
        if item['geometry']['full_rows'] > min(cfg['batch_rows'], cfg['teacher_batch_rows']):
            raise ValueError('training field exceeds configured row budget: ' + str(item['hw']))
        return attach_features(item, build_features)

    def save(step):
        checkpoint = dict(model=model.state_dict(), aligner=aligner.state_dict(), optim=optimizer.state_dict(),
                          ema=ema, step=step, config=cfg, shape=shape, provenance=stamp,
                          parameter_ledger=audit, torch_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all())
        temporary = out / 'ckpt.tmp'
        torch.save(checkpoint, temporary)
        temporary.replace(out / 'ckpt.pt')

    metric_names = ('loss', 'out', 'mse', 'detail', 'feature', 'moe_aux', 'moe_z', 'psnr_teacher')
    with ThreadPoolExecutor(max_workers=cfg['prep_workers']) as cpu_pool, \
            ThreadPoolExecutor(max_workers=1) as teacher_pool, \
            open(out / 'log.csv', 'a', newline='', encoding='utf-8') as log, \
            open(out / 'sampling.jsonl', 'a', encoding='utf-8') as sample_log, \
            open(out / 'routing.jsonl', 'a', encoding='utf-8') as route_log:
        writer = csv.DictWriter(log, fieldnames=['step', 'lr', *metric_names, 'ms'])
        if log.tell() == 0:
            writer.writeheader()
        pending = deque()
        next_step = start
        def enqueue():
            nonlocal next_step
            if next_step < cfg['steps']:
                pending.append([cpu_pool.submit(prepare_item, next_step * cfg['batch'] + bi)
                                for bi in range(cfg['batch'])])
                next_step += 1
        enqueue()
        enqueue()
        future = None
        mode_counts, source_counts = Counter(), Counter()
        last_validation = None
        def collect():
            raw = [f.result() for f in pending.popleft()]
            enqueue()
            return raw
        if start < cfg['steps']:
            future = teacher_pool.submit(teacher.prepare, collect())
        for step in range(start, cfg['steps']):
            started = time.perf_counter()
            raw = future.result()
            mode_counts.update(p['metadata']['mode'] for p in raw)
            source_counts.update(p['metadata']['source'] for p in raw)
            if step + 1 < cfg['steps'] and cfg.get('teacher_overlap', False):
                future = teacher_pool.submit(teacher.prepare, collect())
            lr = losses.lr_at(step, cfg)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            grouped = defaultdict(list)
            for item in raw:
                grouped[item['hw']].append(item)
            totals = dict.fromkeys(metric_names, 0.)
            for group in grouped.values():
                limit = max(1, cfg['batch_rows'] // group[0]['geometry']['full_rows'])
                for begin in range(0, len(group), limit):
                    chunk = group[begin:begin + limit]
                    loss, metrics = chunk_loss(model, aligner, chunk, cfg, helpers)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite loss at step ' + str(step))
                    factor = len(chunk) / cfg['batch']
                    (loss * factor).backward()
                    for key in metric_names:
                        totals[key] += float(metrics[key]) * factor
            torch.nn.utils.clip_grad_norm_(params, cfg['clip'], error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) % cfg['ema_every'] == 0:
                decay = cfg['ema'] ** cfg['ema_every']
                with torch.no_grad():
                    for key, value in model.state_dict().items():
                        ema[key].mul_(decay).add_(value.detach(), alpha=1 - decay)
            if step % cfg['log_every'] == 0 or step + 1 == cfg['steps']:
                row = dict(step=step, lr=lr, **totals, ms=(time.perf_counter() - started) * 1000)
                writer.writerow(row)
                log.flush()
                sampled = dict(step=step, window_modes=dict(mode_counts), window_sources=dict(source_counts),
                               samples=[dict(**p['metadata'], noise_seed=p['noise_seed']) for p in raw])
                sample_log.write(json.dumps(sampled) + '\n')
                sample_log.flush()
                mode_counts.clear()
                source_counts.clear()
                route_log.write(json.dumps(dict(step=step, scope='last_training_microbatch',
                                                layers=model.routing_snapshot())) + '\n')
                route_log.flush()
                print(json.dumps(row), flush=True)
                if experiment:
                    experiment.log_metrics({('psnr/teacher' if k == 'psnr_teacher' else 'train/' + k): v
                                            for k, v in row.items() if k != 'step'}, step=step)
            if cfg['val_every'] and (step + 1) % cfg['val_every'] == 0:
                if cfg.get('teacher_overlap', False) and step + 1 < cfg['steps']:
                    future.result()  # no teacher geometry/budget mutation during eval
                means = validate(model, ema, source, teacher, cfg, build_features, step, out)
                last_validation = step
                if experiment:
                    experiment.log_metrics({'val/' + mode + '/' + key: value[key]
                                            for mode, value in means.items() if value['count']
                                            for key in ('psnr_teacher', 'psnr_input', 'psnr_roi', 'psnr_roi_input')
                                            if key in value}, step=step)
            if cfg['ckpt_every'] and (step + 1) % cfg['ckpt_every'] == 0:
                save(step)
            if step + 1 < cfg['steps'] and not cfg.get('teacher_overlap', False):
                future = teacher_pool.submit(teacher.prepare, collect())
        if start < cfg['steps']:
            if last_validation != cfg['steps'] - 1:
                validate(model, ema, source, teacher, cfg, build_features, cfg['steps'] - 1, out)
            save(cfg['steps'] - 1)
    if experiment:
        experiment.end()
    print('checkpoint:', out / 'ckpt.pt', flush=True)


if __name__ == '__main__':
    main()
