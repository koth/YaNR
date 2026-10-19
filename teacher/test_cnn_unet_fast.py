"""Reference parity, deterministic resize gradients, fusion and package checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F

from nr_cnn_unet import CNNUNet, ExpertFFN
from nr_cnn_unet_fast import (ARCHITECTURE, CNNUNetFast, GroupedExpertFFN,
                              RepConv3, load_fast_shape, resize_bilinear)
from nr_package import load_student_package, write_package
from test_single_frame import small_geometry

ROOT = Path(__file__).resolve().parents[1]


def tiny_shape():
    cfg = load_fast_shape(ROOT / 'shapes/cnn_unet_fast.json')
    cfg.pop('parameter_targets', None)
    for name in cfg['channels']:
        cfg['channels'][name] = 16
        cfg['expert_hidden'][name] = 24
        cfg['blocks_enc'][name] = 1
    for name in cfg['blocks_dec']:
        cfg['blocks_dec'][name] = 1
    return cfg


class FastCNNTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(8)

    def test_grouped_experts_outputs_input_and_parameter_gradients(self):
        for collapsed in (False, True):
            torch.manual_seed(17)
            original = ExpertFFN(16, 24)
            if collapsed:
                with torch.no_grad():
                    original.router.weight.zero_()
                    original.router.bias.zero_()  # first expert on ties; seven empty groups
            grouped = GroupedExpertFFN.from_legacy(original)
            x = torch.randn(2, 16, 7, 11, requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            expected, actual = original(x), grouped(y)
            torch.testing.assert_close(actual, expected, atol=3e-7, rtol=2e-5)
            gradient = torch.randn_like(actual)
            expected.backward(gradient)
            actual.backward(gradient)
            torch.testing.assert_close(x.grad, y.grad, atol=3e-7, rtol=3e-5)
            torch.testing.assert_close(original.router.weight.grad, grouped.router.weight.grad,
                                       atol=2e-6, rtol=3e-5)
            for index, expert in enumerate(original.bank):
                for old, new, transpose in ((expert[0].weight, grouped.w1.grad[index], True),
                                            (expert[0].bias, grouped.b1.grad[index], False),
                                            (expert[2].weight, grouped.w2.grad[index], True),
                                            (expert[2].bias, grouped.b2.grad[index], False)):
                    want = torch.zeros_like(old) if old.grad is None else old.grad
                    torch.testing.assert_close(new.T if transpose else new, want, atol=3e-6, rtol=3e-5)
            if collapsed:
                self.assertEqual(grouped.route_counts.tolist(), [154, 0, 0, 0, 0, 0, 0, 0])

    def test_native_resize_forward_and_ordered_backward(self):
        for source, target in (((7, 11), (13, 19)), ((7, 11), (3, 4)),
                               ((1, 1), (5, 9)), ((9, 7), (1, 1)), ((7, 9), (7, 9))):
            x = torch.randn(2, 3, *source, dtype=torch.float64, requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            actual = resize_bilinear(x, target)
            expected = F.interpolate(y, target, mode='bilinear', align_corners=False)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            gradient = torch.randn_like(actual)
            actual.backward(gradient)
            expected.backward(gradient)
            torch.testing.assert_close(x.grad, y.grad, atol=2e-14, rtol=1e-12)

    def test_tiled_executor_matches_sorted_groups(self):
        torch.manual_seed(18)
        grouped = GroupedExpertFFN(16, 24)
        tiled = copy.deepcopy(grouped)
        tiled.execution = 'tiled'
        x = torch.randn(2, 16, 5, 13, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        expected, actual = grouped(x), tiled(y)
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-5)
        gradient = torch.randn_like(actual)
        expected.backward(gradient)
        actual.backward(gradient)
        torch.testing.assert_close(y.grad, x.grad, atol=5e-7, rtol=3e-5)
        for (_, p), (_, q) in zip(grouped.named_parameters(), tiled.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=4e-6, rtol=3e-5)

    def test_auto_fallback_and_parameter_budget(self):
        grouped = GroupedExpertFFN(16, 24)
        fallback = copy.deepcopy(grouped)
        fallback.execution = 'auto'
        fallback.split_max_rows = 16
        x = torch.randn(1, 16, 7, 11)
        torch.testing.assert_close(fallback(x), grouped(x), atol=2e-7, rtol=2e-5)
        model = CNNUNetFast(load_fast_shape(ROOT / 'shapes/cnn_unet_fast.json'), small_geometry())
        ledger = model.parameter_ledger()
        self.assertEqual(ledger['total'], 56287389)
        self.assertEqual(ledger['deploy_total'], 55358397)

    def test_rep_fusion_nonzero_branches(self):
        module = RepConv3(16).double()
        with torch.no_grad():
            for branch in (module.point, module.horizontal, module.vertical):
                branch.weight.normal_(std=.03)
                branch.bias.normal_(std=.02)
        x = torch.randn(2, 16, 11, 17, dtype=torch.float64)
        expected = module(x)
        module.fuse()
        torch.testing.assert_close(module(x), expected, atol=3e-15, rtol=2e-12)
        self.assertEqual(len(list(module.parameters())), 2)

    def test_legacy_full_network_conversion_and_capture_parity(self):
        cfg = tiny_shape()
        cfg['architecture'] = 'cnn_unet_moe8'
        original = CNNUNet(cfg, small_geometry(32, 24)).eval()
        grouped = CNNUNetFast.from_legacy(original).eval()
        original.capture_enabled = grouped.capture_enabled = True
        x = torch.randn(2 * 32 * 24, 16)
        with torch.no_grad():
            expected = original(x, batch=2)
            actual = grouped(x, batch=2)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=2e-4)
        self.assertEqual(set(original.captures), set(grouped.captures))
        for key in original.captures:
            torch.testing.assert_close(grouped.captures[key], original.captures[key], atol=2e-5, rtol=3e-4)

    def test_training_loss_batch_and_native_export(self):
        torch.manual_seed(11)
        model = CNNUNetFast(tiny_shape(), small_geometry(32, 24))
        x = torch.randn(2 * 32 * 24, 16)
        head = model(x, batch=2)
        aux, z = model.moe_losses()
        (head[:, :3].square().mean() + .01 * aux + .001 * z).backward()
        self.assertGreater(float(model.enc['full'][0].ffn.router.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.enc['full'][0].conv.point.weight.grad.abs().sum()), 0)
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))
        model.eval()
        with torch.no_grad():
            expected = model(x, batch=2)
            fused = copy.deepcopy(model).switch_to_deploy()
            native = fused.to_native_model()
            torch.testing.assert_close(fused(x, batch=2), expected, atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(native(x, batch=2), expected, atol=1e-5, rtol=2e-4)
        self.assertEqual(sum(p.numel() for p in native.parameters()), model.parameter_ledger()['deploy_total'])

    def test_fast_package_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            cfg = tiny_shape()
            shape_file = folder / 'shape.json'
            shape_file.write_text(json.dumps(cfg), encoding='utf-8')
            from nr_geometry import geometry_from_valid
            model = CNNUNetFast(cfg, geometry_from_valid(320, 320))
            write_package(model, model.state_dict(), folder / 'package', shape_file)
            restored, meta = load_student_package(folder / 'package', 320, 320, 'cpu')
            self.assertEqual(meta['model']['architecture'], ARCHITECTURE)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
