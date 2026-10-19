#!/usr/bin/env python3
"""Distillation trainer for the independent CNN U-Net MoE student.

This entry point does not modify or import the training loop of
train_distill.py.  It reuses the shared teacher/data/loss helpers and the
independent model in nr_cnn_unet.py.

Example:
  python3 train_distill_cnn.py \
    --config ../remote/run_cnn_unet.json \
    --data ../data/full/DIV2K_train_HR ... \
    -o runs/cnn_unet
"""
import argparse
import csv
import json
import math
import os
import sys
import threading
import time
try:
    from comet_ml import Experiment
except ImportError:                                                    # pragma: no cover
    Experiment = None

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate import TeacherAnnotator                                   # noqa: E402
from nr_cnn_unet import CNNUNet, load_cnn_shape                         # noqa: E402
from nr_geometry import geometry_from_valid                             # noqa: E402
from nr_history import reproject_history, store_history                 # noqa: E402
from nr_precision import tf32_matmul                                    # noqa: E402
from nr_student import student_alignment                                # noqa: E402
from run_image import build_features                                    # noqa: E402
from train_distill import (DEFAULTS, FeatureAligner, FrameSource,        # noqa: E402
                           comet_key, feature_losses, laplacian_l1, lr_at,
                           output_losses, psnr, sobel_l1)


ORDER = ('full', 'd0', 'd1', 'd2', 'd3', 'd4')
TEACHER_DIMS = {
    's-enc-full': 32, 's-enc-d0': 32, 's-enc-d1': 64, 's-enc-d2': 128,
    's-enc-d3': 256, 's-enc-d4': 512, 's-vit': 1024,
    's-dec-d4': 512, 's-dec-d3': 256, 's-dec-d2': 128, 's-dec-d1': 64,
    's-dec-d0': 32,
}


def level_size(g, name):
    if name == 'full':
        return g['full_width'], g['full_height']
    level = g['levels'][int(name[1:])]
    return level['width'], level['height']


def parse_args():
    p = argparse.ArgumentParser(description='CNN U-Net MoE distillation')
    p.add_argument('--config', help='JSON config (CLI overrides)')
    p.add_argument('--data', required=True, nargs='+', help='image roots')
    p.add_argument('-o', '--out', required=True, help='run directory')
    p.add_argument('--resume', help='checkpoint .pt')
    p.add_argument('--shape', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'shapes', 'cnn_unet_moe8.json'))
    p.add_argument('--teacher-weights', dest='teacher_weights')
    p.add_argument('--size', help='edge length(s), e.g. 320,512')
    p.add_argument('--size-weights', dest='size_weights', help='e.g. 0.7,0.3')
    p.add_argument('--batch', type=int)
    p.add_argument('--batch-rows', dest='batch_rows', type=int)
    p.add_argument('--teacher-batch-rows', dest='teacher_batch_rows', type=int)
    p.add_argument('--prep-workers', dest='prep_workers', type=int)
    p.add_argument('--post-workers', dest='post_workers', type=int)
    p.add_argument('--steps', type=int)
    p.add_argument('--overfit', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--wd', type=float)
    p.add_argument('--warmup', type=int)
    p.add_argument('--ema', type=float)
    p.add_argument('--ema-every', dest='ema_every', type=int)
    p.add_argument('--clip', type=float)
    p.add_argument('--loss-out', dest='loss_out', type=float)
    p.add_argument('--loss-detail', dest='loss_detail', type=float)
    p.add_argument('--loss-feature', dest='loss_feature', type=float)
    p.add_argument('--loss-temporal', dest='loss_temporal', type=float)
    p.add_argument('--loss-logit', dest='loss_logit', type=float)
    p.add_argument('--loss-texture', dest='loss_texture', type=float)
    p.add_argument('--loss-texstat', dest='loss_texstat', type=float)
    p.add_argument('--tex-boost', dest='tex_boost', type=float)
    p.add_argument('--tex-sigma', dest='tex_sigma', type=float)
    p.add_argument('--texstat-boost', dest='texstat_boost', type=float)
    p.add_argument('--texture-floor', dest='texture_floor', type=float)
    p.add_argument('--texture-knee', dest='texture_knee', type=float)
    p.add_argument('--action-floor', dest='action_floor', type=float)
    p.add_argument('--action-knee', dest='action_knee', type=float)
    p.add_argument('--pair-fraction', dest='pair_fraction', type=float)
    p.add_argument('--moe-aux', dest='moe_aux', type=float)
    p.add_argument('--moe-z', dest='moe_z', type=float)
    p.add_argument('--seed', type=int)
    p.add_argument('--log-every', dest='log_every', type=int)
    p.add_argument('--ckpt-every', dest='ckpt_every', type=int)
    p.add_argument('--vis-every', dest='vis_every', type=int)
    p.add_argument('--student-tf32', dest='student_tf32', type=int, choices=(0, 1))
    p.add_argument('--student-compile', dest='student_compile', type=int, choices=(0, 1))
    p.add_argument('--comet-disabled', dest='comet_disabled', action='store_true', default=None)
    return p.parse_args()


def load_cfg(args):
    cfg = dict(DEFAULTS)
    cfg['shape'] = args.shape
    if args.config:
        with open(args.config) as fh:
            cfg.update(json.load(fh))
    # Keep the new entry point independent of experimental keys that may not
    # exist in an older train_distill.DEFAULTS.
    for key, default in (('prep_workers', 4), ('post_workers', 4),
                         ('student_tf32', False), ('student_compile', False),
                         ('moe_aux', 0.01), ('moe_z', 0.001)):
        cfg.setdefault(key, default)
    for key in ('shape', 'teacher_weights', 'batch', 'steps', 'overfit', 'lr', 'wd',
                'warmup', 'ema', 'ema_every', 'clip', 'loss_out', 'loss_detail',
                'loss_feature', 'loss_temporal', 'loss_logit', 'loss_texture',
                'loss_texstat', 'tex_boost', 'tex_sigma', 'texstat_boost',
                'texture_floor', 'texture_knee', 'action_floor', 'action_knee',
                'pair_fraction', 'moe_aux', 'moe_z', 'seed', 'batch_rows',
                'teacher_batch_rows', 'prep_workers', 'post_workers',
                'log_every', 'ckpt_every', 'vis_every', 'student_tf32', 'student_compile',
                'comet_disabled'):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    if args.size is not None:
        cfg['size'] = [int(v) for v in str(args.size).split(',')]
    else:
        cfg['size'] = [int(v) for v in (cfg['size'] if isinstance(cfg['size'], list)
                                        else str(cfg['size']).split(','))]
    if args.size_weights is not None:
        cfg['size_weights'] = args.size_weights
    if cfg.get('size_weights') is not None:
        weights = cfg['size_weights']
        cfg['size_weights'] = [float(v) for v in (weights if isinstance(weights, list)
                                                  else str(weights).split(','))]
        if len(cfg['size_weights']) != len(cfg['size']):
            raise SystemExit('size_weights length must match --size')
    if cfg.get('student_precision', 'fp32') != 'fp32':
        raise SystemExit('CNN U-Net trainer currently requires student_precision=fp32')
    if not cfg.get('teacher_weights'):
        cfg['teacher_weights'] = os.environ.get('NR_WEIGHTS')
    if not cfg.get('teacher_weights'):
        raise SystemExit('no teacher weights: --teacher-weights or NR_WEIGHTS')
    return cfg


def main():
    args = parse_args()
    cfg = load_cfg(args)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        raise SystemExit('CNN U-Net training requires CUDA')

    shape_cfg = load_cnn_shape(cfg['shape'])
    sizes = cfg['size']
    geo_cache = {s: geometry_from_valid(s, s) for s in sizes}
    g0 = geo_cache[sizes[0]]
    model = CNNUNet(shape_cfg, g0).to(device)
    model.capture_enabled = True
    print(f"model: {sum(p.numel() for p in model.parameters()):,} params")
    student_fwd = model
    if cfg.get('student_compile'):
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.cache_size_limit = 1024
        student_fwd = torch.compile(model, dynamic=True)
        print('student: torch.compile(dynamic=True)')

    ch = shape_cfg['channels']
    s_dims = {
        's-enc-full': ch['full'], 's-enc-d0': ch['d0'], 's-enc-d1': ch['d1'],
        's-enc-d2': ch['d2'], 's-enc-d3': ch['d3'], 's-enc-d4': ch['d4'],
        's-vit': ch['d5'], 's-dec-d4': ch['d4'], 's-dec-d3': ch['d3'],
        's-dec-d2': ch['d2'], 's-dec-d1': ch['d1'], 's-dec-d0': ch['d0'],
    }
    align_pairs = {
        name.replace('-', '_'): (s_dims[name], TEACHER_DIMS[name])
        for name in student_alignment() if name != 's-head'
    }
    aligner = FeatureAligner(align_pairs).to(device)
    teacher_caps = tuple(t for t in student_alignment().values() if t != 'head')
    feature_weights = {name: 1.0 for name in student_alignment()}

    params = list(model.parameters()) + list(aligner.parameters())
    optim = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['wd'], fused=True)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    start = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        aligner.load_state_dict(ckpt['aligner'])
        optim.load_state_dict(ckpt['optim'])
        ema = {k: v.to(device) for k, v in ckpt['ema'].items()}
        start = int(ckpt['step']) + 1
        print(f'resumed at step {start}')

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'config.json'), 'w') as fh:
        json.dump(cfg, fh, indent=1, sort_keys=True)

    experiment = None
    key = None if cfg.get('comet_disabled', False) else comet_key(cfg)
    if key and Experiment is not None:
        experiment = Experiment(api_key=key, project_name=cfg['comet_project'],
                                workspace=cfg.get('comet_workspace') or None,
                                log_code=False)
        experiment.log_parameters({k: v for k, v in cfg.items() if k != 'comet_key'})
        print('comet:', experiment.url)
    else:
        print('comet: disabled')

    source = FrameSource(args.data, sizes, cfg, overfit=cfg['overfit'])
    prep_pool = ThreadPoolExecutor(max_workers=max(1, int(cfg['prep_workers'])))
    post_pool = (ThreadPoolExecutor(max_workers=max(1, int(cfg['post_workers'])))
                 if int(cfg['post_workers']) > 1 else None)
    teacher_tls = threading.local()
    teacher_lock = threading.Lock()
    teacher_pool = ThreadPoolExecutor(max_workers=1)
    step_futs = deque()

    def teacher_for(s):
        cache = getattr(teacher_tls, 'teachers', None)
        if cache is None:
            cache = {}
            teacher_tls.teachers = cache
        hit = cache.get(s)
        if hit is None:
            with teacher_lock:
                ann = TeacherAnnotator(cfg['teacher_weights'], geo_cache[s])
            hit = (ann, torch.cuda.Stream())
            cache[s] = hit
        return hit

    def prep_item(step, bi):
        pair = (cfg['pair_fraction'] > 0 and
                np.random.default_rng([cfg['seed'], step, 7]).random() < cfg['pair_fraction'])
        if pair:
            prev_p, cur_p, motion, lane, _ = source.sample_pair(step * cfg['batch'] + bi)
            seed_a = int(np.random.default_rng([cfg['seed'], step, bi, 1]).integers(1 << 30))
            noise_seed = int(np.random.default_rng([cfg['seed'], step, bi, 2]).integers(1 << 30))
            s = cur_p.shape[0]
            g_s = geo_cache[s]
            feats_a = build_features(prev_p, g_s, seed_a, lane['style'], lane['tone'],
                                     lane['structure'], lane['skin'], lane['auto_mask'])
            return dict(kind='pair', s=s, proxy=cur_p, prev=prev_p, motion=motion,
                        lane=lane, noise_seed=noise_seed, feats_a=feats_a)
        proxy, noise_seed, lane, _ = source.sample(step * cfg['batch'] + bi)
        s = proxy.shape[0]
        g_s = geo_cache[s]
        feats = build_features(proxy, g_s, noise_seed, lane['style'], lane['tone'],
                               lane['structure'], lane['skin'], lane['auto_mask'])
        return dict(kind='single', s=s, proxy=proxy, lane=lane,
                    noise_seed=noise_seed, feats=feats, repro=None)

    prep_seq = {'step': start, 'bi': 0}
    prep_futs = deque()
    prep_times = deque()

    def submit_next():
        st, bi = prep_seq['step'], prep_seq['bi']
        if st >= cfg['steps']:
            return False
        prep_seq['bi'] = bi + 1
        if prep_seq['bi'] >= cfg['batch']:
            prep_seq['step'], prep_seq['bi'] = st + 1, 0
        prep_futs.append(prep_pool.submit(prep_item, st, bi))
        return True

    def collect_raw():
        t0 = time.perf_counter()
        raw = [prep_futs.popleft().result() for _ in range(cfg['batch'])]
        for _ in range(cfg['batch']):
            submit_next()
        prep_times.append((time.perf_counter() - t0) * 1000)
        return raw

    for _ in range(cfg['batch'] * 2):
        submit_next()

    def finish_pair(p, h, g_s, s):
        t_neural_a = np.clip(p['prev'] + h[..., :3] / 4, 0, 1).astype(np.float32)
        hist = store_history(t_neural_a)
        repro = reproject_history(hist, p['motion'])
        p['repro'] = repro
        p['t_neural_a'] = t_neural_a
        p['feats'] = build_features(p['proxy'], g_s, p['noise_seed'], p['lane']['style'],
                                    p['lane']['tone'], p['lane']['structure'],
                                    p['lane']['skin'], p['lane']['auto_mask'], history=repro)

    def prepare_step(raw):
        # Previous-frame teacher pass: pairs only.
        by_pair = {}
        for p in raw:
            if p['kind'] == 'pair':
                by_pair.setdefault(p['s'], []).append(p)
        for s, group in by_pair.items():
            g_s = geo_cache[s]
            budget = int(cfg.get('teacher_batch_rows') or cfg['batch_rows'])
            limit = max(1, budget // g_s['full_rows'])
            for c0 in range(0, len(group), limit):
                chunk = group[c0:c0 + limit]
                b = len(chunk)
                ann, stream = teacher_for(s)
                prev_feats = np.stack([p['feats_a'] for p in chunk]).reshape(-1, 16)
                with torch.cuda.stream(stream):
                    res_a = ann.annotate(prev_feats, batch=b)
                heads = res_a['head'].reshape(b, g_s['full_height'], g_s['full_width'], 4)
                tasks = []
                for i, p in enumerate(chunk):
                    if post_pool is None:
                        finish_pair(p, heads[i][:s, :s], g_s, s)
                    else:
                        tasks.append(post_pool.submit(finish_pair, p, heads[i][:s, :s], g_s, s))
                for task in tasks:
                    task.result()

        # Current-frame teacher pass: all samples, grouped by size.
        by_size = {}
        for p in raw:
            by_size.setdefault(p['s'], []).append(p)
        for s, group in by_size.items():
            g_s = geo_cache[s]
            budget = int(cfg.get('teacher_batch_rows') or cfg['batch_rows'])
            limit = max(1, budget // g_s['full_rows'])
            ann, stream = teacher_for(s)
            for c0 in range(0, len(group), limit):
                chunk = group[c0:c0 + limit]
                b = len(chunk)
                feats = np.stack([p['feats'] for p in chunk]).reshape(-1, 16)
                with torch.cuda.stream(stream):
                    res = ann.annotate(feats, capture=teacher_caps, batch=b)
                heads = res['head'].reshape(b, g_s['full_height'], g_s['full_width'], 4)
                for i, p in enumerate(chunk):
                    p['t_head'] = heads[i][:s, :s]
                    p['t_scale'] = ann.blend_scale
                    per = {}
                    for name in teacher_caps:
                        arr = res[name]
                        n = arr.shape[0] // b
                        per[name] = arr[i * n:(i + 1) * n]
                    p['t_res'] = per
        return raw

    log_path = os.path.join(args.out, 'log.csv')
    new_log = not os.path.exists(log_path)
    log = open(log_path, 'a', newline='')
    writer = csv.writer(log)
    if new_log:
        writer.writerow(['step', 'lr', 'loss', 'out', 'detail', 'feature', 'temporal',
                         'psnr_teacher', 'ms', 'prep_ms', 'wait_ms', 'comp_ms', 'moe_aux'])

    def save_ckpt(step):
        torch.save({'model': model.state_dict(), 'aligner': aligner.state_dict(),
                    'optim': optim.state_dict(), 'ema': ema, 'step': step,
                    'config': cfg}, os.path.join(args.out, 'ckpt.pt'))

    step_futs.append(teacher_pool.submit(prepare_step, collect_raw()))

    for step in range(start, cfg['steps']):
        t0 = time.perf_counter()
        lr = lr_at(step, cfg)
        for group in optim.param_groups:
            group['lr'] = lr
        optim.zero_grad()
        t_wait0 = time.perf_counter()
        raw = step_futs.popleft().result()
        wait_ms = (time.perf_counter() - t_wait0) * 1000
        prep_ms = prep_times.popleft()
        if step + 1 < cfg['steps']:
            step_futs.append(teacher_pool.submit(prepare_step, collect_raw()))
        t_comp0 = time.perf_counter()
        by_kind = {}
        for p in raw:
            by_kind.setdefault((p['s'], p['kind']), []).append(p)

        total_n = 0
        metrics = dict(out=0., detail=0., feature=0., temporal=0., psnr=0., moe=0.)
        for (s, kind), group in by_kind.items():
            g_s = geo_cache[s]
            model.g = g_s
            max_b = max(1, int(cfg['batch_rows']) // g_s['full_rows'])
            for c0 in range(0, len(group), max_b):
                chunk = group[c0:c0 + max_b]
                b = len(chunk)
                total_n += b
                x = torch.from_numpy(np.stack([p['feats'] for p in chunk])).reshape(-1, 16).to(device)
                with tf32_matmul(cfg.get('student_tf32', False)):
                    s_head = student_fwd(x, batch=b)
                s_head = s_head.view(b, g_s['full_height'], g_s['full_width'], 4)[:, :s, :s].float()

                t_res_stack = {}
                for name in teacher_caps:
                    arr = np.stack([p['t_res'][name] for p in chunk])
                    t_res_stack[name] = arr.reshape(-1, arr.shape[-1])
                feat, _ = feature_losses(model.captures, t_res_stack, aligner, feature_weights,
                                         diagnostics=False)
                moe_aux = model.moe_aux_loss() if cfg['moe_aux'] else s_head.new_zeros(())
                moe_z = model.moe_z_loss() if cfg['moe_z'] else s_head.new_zeros(())

                s_neural_a = None
                if kind == 'pair' and cfg['loss_temporal'] > 0:
                    xa = torch.from_numpy(np.stack([p['feats_a'] for p in chunk])).reshape(-1, 16).to(device)
                    with torch.no_grad(), tf32_matmul(cfg.get('student_tf32', False)):
                        sa = student_fwd(xa, batch=b)
                    sa = sa.view(b, g_s['full_height'], g_s['full_width'], 4)[:, :s, :s].float()
                    prev_t = torch.from_numpy(np.stack([p['prev'] for p in chunk])).to(device)
                    s_neural_a = torch.clamp(prev_t + sa[..., :3] / 4.0, 0, 1)

                proxy_t = torch.from_numpy(np.stack([p['proxy'] for p in chunk])).to(device)
                t_head_t = torch.from_numpy(np.stack([p['t_head'] for p in chunk])).to(device)
                repro_t = (torch.from_numpy(np.stack([p['repro'] for p in chunk])).to(device)
                           if kind == 'pair' else None)
                t_scale = torch.tensor([p['t_scale'] for p in chunk], device=device)
                out_v, s_neural, t_neural, _, _, _ = output_losses(
                    s_head, t_head_t, proxy_t, repro=repro_t, s_scale=model.blend_scale,
                    t_scale=t_scale, logit_w=cfg['loss_logit'], act_floor=cfg['action_floor'],
                    act_knee=cfg['action_knee'], tex_w=cfg['loss_texture'],
                    tex_floor=cfg['texture_floor'], tex_knee=cfg['texture_knee'],
                    tex_boost=cfg['tex_boost'], tex_sigma=cfg['tex_sigma'],
                    texstat_w=cfg['loss_texstat'], texstat_boost=cfg['texstat_boost'])
                detail_v = laplacian_l1(s_neural, t_neural) + sobel_l1(s_neural, t_neural)
                temporal_v = s_head.new_zeros(b)
                if s_neural_a is not None:
                    t_na = torch.from_numpy(np.stack([p['t_neural_a'] for p in chunk])).to(device)
                    temporal_v = (s_neural - s_neural_a - (t_neural - t_na)).abs().reshape(b, -1).mean(1)
                loss = (cfg['loss_out'] * out_v + cfg['loss_detail'] * detail_v
                        + cfg['loss_temporal'] * temporal_v).mean()
                loss = loss + cfg['loss_feature'] * feat
                loss = loss + cfg['moe_aux'] * moe_aux + cfg['moe_z'] * moe_z
                if not torch.isfinite(loss):
                    print(f'step {step}: non-finite loss, skipping chunk')
                    continue
                (loss * (b / cfg['batch'])).backward()

                metrics['out'] += float(out_v.detach().mean()) * b
                metrics['detail'] += float(detail_v.detach().mean()) * b
                metrics['feature'] += float(feat.detach()) * b
                metrics['temporal'] += float(temporal_v.detach().mean()) * b
                metrics['psnr'] += psnr(s_neural.detach(), t_neural.detach()) * b
                metrics['moe'] += float(moe_aux.detach()) * b

        torch.nn.utils.clip_grad_norm_(params, cfg['clip'])
        optim.step()
        comp_ms = (time.perf_counter() - t_comp0) * 1000
        if cfg['ema'] > 0 and (step + 1) % max(1, cfg['ema_every']) == 0:
            decay = min(cfg['ema'], (1.0 + step) / (100.0 + step))
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(decay).add_(v.detach(), alpha=1 - decay)

        ms = (time.perf_counter() - t0) * 1000
        if step % cfg['log_every'] == 0 or step == cfg['steps'] - 1:
            n = max(1, total_n)
            total_loss = (metrics['out'] / n + metrics['detail'] / n
                          + metrics['feature'] / n + metrics['temporal'] / n)
            row = [step, f'{lr:.2e}',
                   f'{total_loss:.5f}',
                   f"{metrics['out'] / n:.5f}", f"{metrics['detail'] / n:.5f}",
                   f"{metrics['feature'] / n:.5f}", f"{metrics['temporal'] / n:.5f}",
                   f"{metrics['psnr'] / n:.2f}", f'{ms:.0f}', f'{prep_ms:.0f}',
                   f'{wait_ms:.0f}', f'{comp_ms:.0f}', f"{metrics['moe'] / n:.4f}"]
            writer.writerow(row)
            log.flush()
            print(f"step {step:6d}  lr {lr:.2e}  out {metrics['out'] / n:.5f}  "
                  f"detail {metrics['detail'] / n:.5f}  feat {metrics['feature'] / n:.5f}  "
                  f"temp {metrics['temporal'] / n:.5f}  psnr(t) {metrics['psnr'] / n:.2f}  "
                  f"moe {metrics['moe'] / n:.3f}  {ms:.0f} ms  "
                  f"(prep {prep_ms:.0f} wait {wait_ms:.0f} comp {comp_ms:.0f})", flush=True)
            if experiment is not None:
                experiment.log_metrics({
                    'loss/out': metrics['out'] / n, 'loss/detail': metrics['detail'] / n,
                    'loss/feature': metrics['feature'] / n,
                    'loss/temporal': metrics['temporal'] / n,
                    'loss/total': total_loss, 'psnr/teacher': metrics['psnr'] / n,
                    'moe/aux': metrics['moe'] / n, 'train/lr': lr, 'train/ms': ms,
                    'train/prep_ms': prep_ms, 'train/wait_ms': wait_ms,
                    'train/comp_ms': comp_ms,
                }, step=step)
        if cfg['ckpt_every'] and (step + 1) % cfg['ckpt_every'] == 0:
            save_ckpt(step)
            print(f'checkpoint -> {os.path.join(args.out, "ckpt.pt")}', flush=True)

    save_ckpt(cfg['steps'] - 1)
    log.close()
    print('done ->', os.path.join(args.out, 'ckpt.pt'))


if __name__ == '__main__':
    raise SystemExit(main())
