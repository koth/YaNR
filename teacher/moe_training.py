"""Training-only FP32 routing controller, telemetry, EMA/bias and replay state.

Loads count actual routed field rows (including geometry padding), never shared
expert rows or GEMM alignment. A controller is attached only by the trainer, so
eval/inference never update bias or retain microbatch counters.
"""
import copy
import random

import numpy as np
import torch

from nr_shared_moe import RoutedModule


def configure(cfg, shape, explicit=()):
    from moe_shape import is_shared
    if not is_shared(shape):
        return cfg
    for key, default in dict(student_precision='fp32', router_jitter=0., moe_aux=0., moe_z=0.,
                             moe_bias_rate=1e-3, moe_z_formula='logsumexp-square-v1',
                             ema_bias_policy='copy-after-optimizer-controller-v1').items():
        if key not in explicit:
            cfg[key] = default
    if cfg['student_precision'] != 'fp32' or cfg['router_jitter'] != 0 or cfg.get('qat', False):
        raise ValueError('shared-routed-moe-v1 first training implementation requires explicit FP32, no jitter/QAT')
    if cfg['moe_z_formula'] != 'logsumexp-square-v1' or cfg['ema_bias_policy'] != 'copy-after-optimizer-controller-v1':
        raise ValueError('incompatible shared/routed z-loss or EMA bias policy')
    if cfg['moe_bias_rate'] < 0:
        raise ValueError('moe_bias_rate must be nonnegative')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return cfg


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


class BiasController:
    VERSION = 'row-load-step-bias-v1'

    def __init__(self, model, rate=1e-3):
        self.modules = {n: m for n, m in model.named_modules() if isinstance(m, RoutedModule)}
        if not self.modules:
            raise ValueError('bias controller requires shared/routed modules')
        self.rate = float(rate)
        self.pending = {n: torch.zeros(16, dtype=torch.int64, device=m.router_bias.device) for n, m in self.modules.items()}
        self.last_active_step = {n: torch.full((16,), -1, dtype=torch.int64, device=m.router_bias.device) for n, m in self.modules.items()}
        self.updated_steps = {n: torch.zeros(16, dtype=torch.int64, device=m.router_bias.device) for n, m in self.modules.items()}
        self.step = None; self.last_applied_step = -1; self.microbatches = 0
        self.telemetry = {}; self.last_load = {}
        self.regularizer_gradients = {}
        self.gradient_summary = {}

    def begin(self, step):
        if self.step is not None or step <= self.last_applied_step:
            raise ValueError('bias step is already pending or has been applied')
        if any(bool(v.any()) for v in self.pending.values()):
            raise ValueError('unconsumed microbatch loads')
        self.step = int(step); self.microbatches = 0; self.telemetry = {}
        self.regularizer_gradients = {}; self.gradient_summary = {}

    def observe_regularizer(self, loss):
        if not torch.is_tensor(loss) or not loss.requires_grad:
            return
        names = list(self.modules)
        parameters = [self.modules[n].router[1].weight for n in names]
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        for name, gradient in zip(names, grads):
            if gradient is not None:
                if name not in self.regularizer_gradients:
                    self.regularizer_gradients[name] = gradient.detach().clone()
                else:
                    self.regularizer_gradients[name].add_(gradient.detach())

    def capture_gradients(self):
        for name, module in self.modules.items():
            total = module.router[1].weight.grad
            if total is None:
                continue
            regularizer = self.regularizer_gradients.get(name)
            task = total if regularizer is None else total - regularizer
            self.gradient_summary[name] = dict(router_task_gradient_norm=float(task.norm()),
                                                router_regularizer_gradient_norm=0. if regularizer is None else float(regularizer.norm()),
                                                router_total_gradient_norm_before_clip=float(total.norm()))

    def collect(self):
        """Call once after each task forward, before any no_grad temporal pass."""
        if self.step is None:
            raise ValueError('call begin(step) before accumulating loads')
        for name, module in self.modules.items():
            for assign, _probs, _logits in module.moe_stats:
                self.pending[name].add_(torch.bincount(assign, minlength=16))
            for stats in module.training_metric_records:
                record = self.telemetry.setdefault(name, dict(calls=0))
                record['calls'] += 1
                for key, value in stats.items():
                    record[key] = record.get(key, 0.) + float(value.detach())
            module.training_metric_records.clear()
        self.microbatches += 1

    @torch.no_grad()
    def apply(self, step):
        if self.step != step or step <= self.last_applied_step or self.microbatches == 0:
            raise ValueError('missing/duplicate/out-of-order optimizer bias update')
        for name, module in self.modules.items():
            load = self.pending[name]
            if not bool(load.sum()):
                raise ValueError('module received no task rows: ' + name)
            before = module.router_bias.clone()
            module.router_bias.add_(self.rate * (load.sum().float() / 16 - load.float()).sign())
            module.router_bias.sub_(module.router_bias.mean())
            self.last_load[name] = load.clone()
            active = load > 0
            self.last_active_step[name][active] = step
            self.updated_steps[name].add_(active.to(torch.int64))
            self.telemetry.setdefault(name, dict(calls=0))['bias_change_max'] = float((module.router_bias - before).abs().max())
            load.zero_()
        self.last_applied_step = int(step); self.step = None

    def diagnostics(self, model):
        """Per-layer state plus task/regularizer gradient labels; caller logs JSONL."""
        output = {}
        for name, module in self.modules.items():
            record = self.telemetry.get(name, {})
            calls = max(1, record.get('calls', 0))
            item = {k: v / calls for k, v in record.items() if k not in ('calls', 'bias_change_max')}
            item['bias_change_max'] = record.get('bias_change_max', 0.)
            item['load_rows'] = self.last_load.get(name, self.pending[name]).tolist()
            item['bias'] = module.router_bias.detach().cpu().tolist()
            item['last_active_step'] = self.last_active_step[name].tolist()
            item['active_optimizer_steps'] = self.updated_steps[name].tolist()
            item['router_gradient_norm'] = float(module.router[1].weight.grad.norm()) if module.router[1].weight.grad is not None else 0.
            item.update(self.gradient_summary.get(name, {}))
            item['expert_gradient_norms'] = {}
            for key, parameter in module.named_parameters(recurse=False):
                if parameter.ndim >= 2 and parameter.shape[0] == 17 and parameter.grad is not None:
                    item['expert_gradient_norms'][key] = parameter.grad.detach().flatten(1).norm(dim=1).cpu().tolist()
            output[name] = item
        return output

    def state_dict(self):
        return dict(version=self.VERSION, rate=self.rate, step=self.step, last_applied_step=self.last_applied_step,
                    microbatches=self.microbatches, telemetry=copy.deepcopy(self.telemetry),
                    regularizer_gradients={n: v.clone() for n, v in self.regularizer_gradients.items()},
                    pending={n: v.clone() for n, v in self.pending.items()},
                    last_active_step={n: v.clone() for n, v in self.last_active_step.items()},
                    updated_steps={n: v.clone() for n, v in self.updated_steps.items()})

    def load_state_dict(self, state):
        if state['version'] != self.VERSION or state['rate'] != self.rate or set(state['pending']) != set(self.modules):
            raise ValueError('incompatible bias controller checkpoint')
        for key in ('pending', 'last_active_step', 'updated_steps'):
            for name, value in state[key].items():
                getattr(self, key)[name].copy_(value)
        self.step = state['step']; self.last_applied_step = state['last_applied_step']
        self.microbatches = state['microbatches']; self.telemetry = copy.deepcopy(state.get('telemetry', {}))
        self.regularizer_gradients = {n: v.to(self.modules[n].router_bias.device).clone() for n, v in state.get('regularizer_gradients', {}).items()}
        if self.step is None and any(bool(v.any()) for v in self.pending.values()):
            raise ValueError('controller has pending rows without a pending optimizer step')
        if self.step is not None and self.step <= self.last_applied_step:
            raise ValueError('controller checkpoint would repeat an already applied update')


@torch.no_grad()
def update_ema(ema, state, decay):
    """EMA floats for weights; bias snapshots copied AFTER controller update.

Bias is a control state, not a trainable parameter. Arithmetic EMA of bias
would create a controller state which never existed at an update boundary.
"""
    for name, value in state.items():
        if name.endswith('.router_bias') or not value.dtype.is_floating_point:
            ema[name].copy_(value)
        else:
            ema[name].mul_(decay).add_(value.detach(), alpha=1 - decay)


def training_checkpoint(model, aligner, optimizer, ema, cfg, step, controller=None):
    result = dict(step=step, model=model.state_dict(), aligner=aligner.state_dict(),
                  optim=optimizer.state_dict(), ema=ema, config=cfg)
    if controller is not None:
        result.update(routing_controller=controller.state_dict(), rng=rng_state(),
                      ema_bias_source=dict(policy=cfg['ema_bias_policy'], optimizer_step=controller.last_applied_step),
                      source=cfg.get('source', {}))
        if controller.step is not None:
            result['gradients'] = {prefix + n: p.grad.detach().clone()
                                   for prefix, module in [('model.', model), ('aligner.', aligner)]
                                   for n, p in module.named_parameters() if p.grad is not None}
    return result


def restore_gradients(checkpoint, model, aligner):
    for prefix, module in [('model.', model), ('aligner.', aligner)]:
        for name, p in module.named_parameters():
            value = checkpoint.get('gradients', {}).get(prefix + name)
            p.grad = None if value is None else value.to(p.device).clone()


def provenance(shape_path, teacher_dir):
    import hashlib
    from pathlib import Path
    import subprocess
    root = Path(__file__).resolve().parent.parent
    def digest(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=True)
    teacher_files = ['nr_model.py', 'nr_network.py', 'nr_kernels.cu', 'nr_torch.py']
    student_files = ['nr_student.py', 'nr_shared_moe.py', 'moe_training.py', 'train_distill.py']
    return dict(student_commit=result.stdout.strip(), teacher_commit=result.stdout.strip(),
                shape_sha256=digest(shape_path), teacher_manifest_sha256=digest(Path(teacher_dir) / 'manifest.json'),
                teacher_source_sha256={n: digest(root / 'teacher' / n) for n in teacher_files},
                student_source_sha256={n: digest(root / 'teacher' / n) for n in student_files})
