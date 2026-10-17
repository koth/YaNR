"""Run the real training loop with observational step timing; no checkpoint I/O.

Use only for short probes. The production trainer is imported without edits.
Each reported step includes CPU preparation waits, teacher overlap, losses,
backward, optimizer/controller/EMA and ordinary logging overhead.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path.cwd())
p.add_argument('--config', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--steps', type=int, default=40)
p.add_argument('--rows', type=int, default=1179648)
p.add_argument('--teacher-rows', type=int, default=589824)
p.add_argument('--compile', action='store_true')
p.add_argument('--serial', action='store_true')
p.add_argument('--resume')
p.add_argument('--size')
p.add_argument('--pair-fraction', type=float)
p.add_argument('--log-every', type=int, default=10)
p.add_argument('--data', nargs='+', required=True)
args = p.parse_args()
args.root = args.root.resolve()
args.out = args.out.resolve()
if args.out.exists() and any(args.out.iterdir()):
    raise SystemExit('--out must be a new or empty benchmark directory')
args.data = [str(Path(root).resolve()) for root in args.data]
if args.resume:
    args.resume = str(Path(args.resume).resolve())
args.out.mkdir(parents=True, exist_ok=True)
cfg = json.loads(args.config.read_text())
cfg.update(batch_rows=args.rows, teacher_batch_rows=args.teacher_rows,
           student_compile=args.compile, teacher_overlap=not args.serial,
           comet_disabled=True, vis_every=0, ckpt_every=0,
           log_every=args.log_every, steps=args.steps)
if args.size:
    cfg.update(size=args.size, size_weights=None)
if args.pair_fraction is not None:
    cfg['pair_fraction'] = args.pair_fraction
(args.out / 'input.json').write_text(json.dumps(cfg, indent=2) + '\n')
os.chdir(args.root)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(args.root / 'teacher'))
import torch
import PIL
torch.set_num_threads(8)
import train_distill as train

report = dict(root=str(args.root), overrides=vars(args).copy(),
              python=sys.version.split()[0], torch=torch.__version__,
              numpy=train.np.__version__, pillow=PIL.__version__,
              cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
              precision='FP32; TF32 off', step_ms=[])
report['overrides'] = {k: str(v) if isinstance(v, Path) else v for k, v in report['overrides'].items()}
real_range = range

def observed_steps(it):
    for step in it:
        t0 = time.perf_counter()
        yield step
        elapsed = (time.perf_counter() - t0) * 1000
        row = dict(step=step, ms=elapsed, allocated=torch.cuda.memory_allocated(),
                   peak_allocated=torch.cuda.max_memory_allocated(),
                   peak_reserved=torch.cuda.max_memory_reserved())
        report['step_ms'].append(row)
        print('PROBE_STEP ' + json.dumps(row), flush=True)

def observed_range(*values):
    it = real_range(*values)
    if len(values) == 2 and values[1] == cfg['steps'] and sys._getframe(1).f_code is train.main.__code__:
        return observed_steps(it)
    return it

train.range = observed_range
real_save = torch.save

def inspect_checkpoint(state, path, *save_args, **save_kwargs):
    # Compiler/cache internals may also use torch.save. Only replace the
    # trainer's final checkpoint, leaving those unrelated saves intact.
    if not (isinstance(state, dict) and 'model' in state and 'step' in state and
            isinstance(path, (str, os.PathLike)) and
            Path(path).resolve() == args.out / 'ckpt.pt'):
        return real_save(state, path, *save_args, **save_kwargs)
    tensors = list(state['model'].values())
    report['finite_model'] = bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all())
    report['checkpoint_step'] = state['step']
    report['controller_last_step'] = state.get('routing_controller', {}).get('last_applied_step')
    report['tf32_matmul'] = torch.backends.cuda.matmul.allow_tf32
    report['deterministic'] = torch.are_deterministic_algorithms_enabled()
    report['source'] = state.get('source', {})
    print('PROBE_CHECKPOINT ' + json.dumps({k: v for k, v in report.items() if k not in ('step_ms', 'overrides')}), flush=True)
    if not report['finite_model']:
        raise RuntimeError('benchmark ended with non-finite model state')

torch.save = inspect_checkpoint
sys.argv = [str(args.root / 'teacher/train_distill.py'), '--config', str(args.out / 'input.json'),
            '--data', *args.data, '-o', str(args.out)]
if args.resume:
    sys.argv += ['--resume', args.resume]
t0 = time.perf_counter()
try:
    report['exit_code'] = train.main()
except BaseException as exc:
    report['error'] = repr(exc)
    raise
finally:
    report['wall_seconds'] = time.perf_counter() - t0
    report['peak_allocated'] = torch.cuda.max_memory_allocated()
    report['peak_reserved'] = torch.cuda.max_memory_reserved()
    rows = report['step_ms'][5:]
    if rows:
        vals = [r['ms'] for r in rows]
        report.update(steady_mean_ms=statistics.mean(vals), steady_median_ms=statistics.median(vals))
    (args.out / 'probe.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PROBE_RESULT ' + json.dumps({k: v for k, v in report.items() if k != 'step_ms'}), flush=True)
