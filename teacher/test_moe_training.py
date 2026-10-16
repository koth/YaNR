"""Optimizer-boundary, pending-microbatch replay, precision and EMA fixtures."""
import copy
import unittest

import torch
from torch import nn

from moe_shape import candidate
from moe_training import (BiasController, configure, rng_state, restore_rng,
                          training_checkpoint, restore_gradients, update_ema)
from nr_shared_moe import SharedRoutedLinear


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = SharedRoutedLinear(32, 32)
        self.block.training_metrics_enabled = True

    def forward(self, x):
        self.block.moe_stats.clear()
        return self.block(x, 9, 7)


class ControllerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_defaults_and_fp32_router(self):
        shape = candidate([32] * 6, [32] * 6, 32, 32)
        cfg = configure({}, shape)
        self.assertEqual(cfg['student_precision'], 'fp32')
        self.assertEqual(cfg['moe_z'], 0)
        with self.assertRaises(ValueError):
            configure(dict(student_precision='bf16'), shape, {'student_precision'})
        block = SharedRoutedLinear(32, 32)
        seen = []
        hook = block.router[1].register_forward_pre_hook(lambda _m, x: seen.append(x[0].dtype))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            block(torch.randn(63, 32), 9, 7)
        hook.remove()
        self.assertEqual(seen, [torch.float32])

    def test_row_counts_boundary_update_once_eval_freeze(self):
        model = Tiny()
        with torch.no_grad():
            model.block.router[1].weight.zero_()
        controller = BiasController(model, .001)
        controller.begin(0)
        for _ in range(2):
            model(torch.randn(63, 32)); controller.collect()
        self.assertEqual(controller.pending['block'].tolist(), [126] + [0] * 15)
        self.assertTrue(torch.equal(model.block.router_bias, torch.zeros(16)))
        controller.apply(0)
        expected = torch.tensor([-.001875] + [.000125] * 15)
        torch.testing.assert_close(model.block.router_bias, expected, rtol=0, atol=2e-10)
        with self.assertRaises(ValueError):
            controller.apply(0)
        before = model.block.router_bias.clone()
        model.eval(); model(torch.randn(63, 32))
        self.assertTrue(torch.equal(before, model.block.router_bias))
        self.assertEqual(controller.last_applied_step, 0)

    def test_microbatch_resume_gradients_rng_and_ema(self):
        model = Tiny(); aligner = nn.Linear(1, 1)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        controller = BiasController(model)
        ema = {n: p.detach().clone() for n, p in model.state_dict().items()}
        cfg = {'ema_bias_policy': 'copy-after-optimizer-controller-v1'}
        controller.begin(0)
        a, b = torch.randn(63, 32), torch.randn(63, 32)
        model(a).square().mean().backward(); controller.collect()
        checkpoint = copy.deepcopy(training_checkpoint(model, aligner, optim, ema, cfg, -1, controller))

        def finish(m, o, c, e):
            m(b).square().mean().backward(); c.collect()
            o.step(); c.apply(0); update_ema(e, m.state_dict(), .9)

        finish(model, optim, controller, ema)
        restored = Tiny(); restored.load_state_dict(checkpoint['model'])
        optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        optimizer.load_state_dict(checkpoint['optim'])
        replay = BiasController(restored); replay.load_state_dict(checkpoint['routing_controller'])
        restore_gradients(checkpoint, restored, aligner)
        restore_rng(checkpoint['rng'])
        replay_ema = copy.deepcopy(checkpoint['ema'])
        finish(restored, optimizer, replay, replay_ema)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)
            self.assertTrue(torch.equal(ema[name], replay_ema[name]), name)
        self.assertTrue(torch.equal(ema['block.router_bias'], model.block.router_bias))
        self.assertEqual(replay.last_applied_step, 0)
        self.assertEqual(replay.microbatches, 2)
        self.assertTrue(torch.equal(controller.last_load['block'], replay.last_load['block']))
        # RNG capture restores torch's next draw; numpy/python included in state.
        state = rng_state(); first = torch.rand(7); restore_rng(state)
        self.assertTrue(torch.equal(first, torch.rand(7)))

    def test_gradient_sources_and_texture_learnability(self):
        model = Tiny(); controller = BiasController(model, rate=0.)
        x = torch.randn(63, 32)
        # Deterministic synthetic high-frequency residual encoded in an input
        # lane; this tests learnability, not resemblance to a real teacher image.
        texture = torch.tensor([(-1.) ** (i + i // 9) for i in range(63)])[:, None]
        x[:, :1] = texture
        target = texture.expand(-1, 32) * .1
        optimizer = torch.optim.AdamW(model.parameters(), lr=.015, weight_decay=0.)
        initial = float((model(x) - target).square().mean().detach())
        for step in range(24):
            optimizer.zero_grad(); controller.begin(step)
            out = model(x)
            controller.collect()
            task = (out - target).square().mean()
            regularizer = .0001 * model.block.moe_stats[0][2].logsumexp(-1).square().mean()
            controller.observe_regularizer(regularizer)
            (task + regularizer).backward(); controller.capture_gradients()
            optimizer.step(); controller.apply(step)
        final = float((model(x) - target).square().mean().detach())
        self.assertLess(final, initial * .35)
        gradient = controller.gradient_summary['block']
        self.assertGreater(gradient['router_task_gradient_norm'], 0.)
        self.assertGreater(gradient['router_regularizer_gradient_norm'], 0.)


if __name__ == '__main__':
    unittest.main()
