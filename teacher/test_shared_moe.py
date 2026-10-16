"""Small semantic fixtures, independent of trained weights or CUDA availability."""
import copy
import unittest

import torch

from audit_moe import audit_manifest
from moe_shape import candidate, ledger, validate
from moe_cost import account_moe
from nr_geometry import geometry_from_valid
from nr_shared_moe import select, SharedRoutedBlock, SharedRoutedLinear


class SharedMoETest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)

    def test_contract_rejects_semantic_changes(self):
        s = candidate([32] * 6, [32] * 6, 32, 32)
        for key, bad in [('gate', 'renormalized-top1'), ('jitter', 0.1), ('routed_experts', 15)]:
            changed = copy.deepcopy(s); changed['routing'][key] = bad
            with self.assertRaises(ValueError):
                validate(changed)

    def test_tie_bias_and_unbiased_gate(self):
        logits = torch.zeros(3, 16, requires_grad=True)
        with torch.no_grad():
            logits[1, 7] = 1e-6; logits[2, 15] = 2
        pick, gate, _ = select(logits, torch.zeros(16))
        self.assertEqual(pick.tolist(), [0, 7, 15])
        bias = torch.zeros(16); bias[4] = 3
        pick, biased_gate, _ = select(logits, bias)
        self.assertEqual(pick.tolist(), [4, 4, 4])
        self.assertEqual(float(biased_gate[0]), 1.)
        self.assertLess(float(biased_gate[2]), 1.)
        gate.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)

    def test_task_gradient_and_independent_ffn(self):
        block = SharedRoutedBlock(32, 32, attn=False).train()
        x = torch.randn(7 * 9, 32)
        out = block(x, 9, 7)
        target = torch.linspace(-1, 1, out.numel()).reshape_as(out)
        (out * target).mean().backward()
        self.assertGreater(float(block.router[1].weight.grad.abs().sum()), 1e-7)
        self.assertGreater(float(block.expand.grad[0].abs().sum()), 0.)
        self.assertGreater(float(block.expand.grad[1:].abs().sum()), 0.)
        self.assertIsNone(block.router_bias.grad)
        block.eval(); block.capture_enabled = True
        out = block(x, 9, 7)
        assign = block.captures['assign']; gate = block.captures['gate']
        # Explicit per-row, separate nonlinear expert branches.
        rows = []
        for i in range(x.shape[0]):
            e = int(assign[i]) + 1
            wi = (i // 9 // 8) * 2 + (i % 9 // 8)
            shared = torch.nn.functional.silu(x[i] @ block.expand[0]) @ block.narrow[0]
            routed = torch.nn.functional.silu(x[i] @ block.expand[e]) @ block.narrow[e]
            rows.append(x[i] * block.aux_ffn + 0.5 * (shared + gate[wi] * routed))
        torch.testing.assert_close(out, torch.stack(rows), rtol=1e-5, atol=5e-7)

    def test_batch_and_all_hot_small_deep_group(self):
        layer = SharedRoutedLinear(32, 64).eval()
        with torch.no_grad():
            layer.router[1].weight.zero_()
        a, b = torch.randn(4 * 5, 32), torch.randn(4 * 5, 32)
        layer.capture_enabled = True
        out = layer(a, 5, 4)
        self.assertEqual(layer.captures['pick'].tolist(), [0])
        pick = layer.captures['pick'].clone()
        expected = ((a @ layer.weight[0] + layer.bias[0]) + (a @ layer.weight[1] + layer.bias[1])) * 0.5
        torch.testing.assert_close(out, expected, atol=1e-7, rtol=1e-6)
        batch_out = layer(torch.cat([a, b]), 5, 4, 2)
        self.assertTrue(torch.equal(layer.captures['pick'][:1], pick))
        torch.testing.assert_close(batch_out[:20], out, rtol=1e-5, atol=5e-7)

    def test_ledger_budget_and_missing_transitions(self):
        s = candidate([32, 32, 64, 96, 192, 192], [32, 32, 64, 64, 64, 64], 384, 768)
        entries = ledger(s)
        a = audit_manifest({'tensors': entries}, s)
        self.assertEqual(a['totals']['parameters'], 80_651_059)
        self.assertEqual(a['totals']['buffers'], 480)
        self.assertTrue(a['budget']['pass'])
        incomplete = [e for e in entries if e['name'] != 'vit_out.weight']
        self.assertFalse(audit_manifest({'tensors': incomplete}, s)['budget']['transitions_pass'])
        malformed = copy.deepcopy(entries)
        entry = next(e for e in malformed if e['name'] == 'vit_out.weight')
        entry['shape'][0] = 16; entry['byteLength'] = 4 * 16 * 384 * 192
        with self.assertRaises(ValueError):
            audit_manifest({'tensors': malformed}, s)

    def test_cost_geometry_and_alignment(self):
        s = candidate([32] * 6, [32] * 6, 32, 32)
        g = geometry_from_valid(137, 201)
        result = account_moe(s, g)
        rows = {l['name']: l for l in result['layers']}
        f = rows['enc_blocks.full.0']
        self.assertEqual(f['linear_macs'], f['rows'] * 4 * 32 * 32)
        self.assertEqual(f['attention_padded_reference_macs'], 0)
        self.assertEqual(rows['dec_trans.d0']['rows'], g['levels'][1]['rows'])
        self.assertIsNone(f['alignment_extra_macs'])
        measured = account_moe(s, g, {'enc_blocks.full.0': [f['rows']] + [0] * 15})
        self.assertEqual(measured['layers'][0]['alignment_extra_rows'], (-f['rows']) % 32)


if __name__ == '__main__':
    unittest.main()
