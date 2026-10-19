"""CPU semantic checks for single-frame data, routing, training and export."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('NR_DEVICE', 'cpu')
os.environ.setdefault('NR_TRITON', '0')
os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np
from PIL import Image
import torch

from nr_cnn_single import PixelNorm, SingleFrameCNN, Top1FFN, load_single_shape, resize_bilinear
from nr_geometry import geometry_from_valid
from single_frame_data import SingleFrameSource, build_manifest, read_rgb
from train_distill_single import (ROOT, TEACHER_DIMS, attach_features, chunk_loss,
                                  load_config, load_helpers, make_aligner, validate)


def tiny_shape():
    shape = load_single_shape(ROOT / 'shapes/cnn_single_56m.json')
    shape.pop('parameter_targets')
    for name in shape['channels']:
        shape['channels'][name] = 8
        shape['expert_hidden'][name] = 16
        shape['blocks_enc'][name] = 1
    for name in shape['blocks_dec']:
        shape['blocks_dec'][name] = 1
    return shape


def small_geometry(w=64, h=48):
    levels = []
    lw, lh = w, h
    for _ in range(6):
        lw, lh = max(1, (lw + 1) // 2), max(1, (lh + 1) // 2)
        levels.append(dict(width=lw, height=lh, rows=lw * lh))
    return dict(full_width=w, full_height=h, full_rows=w * h,
                valid_width=w, valid_height=h, levels=levels)


class SingleModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(8)

    def test_production_parameter_budget(self):
        shape = load_single_shape(ROOT / 'shapes/cnn_single_56m.json')
        model = SingleFrameCNN(shape, geometry_from_valid(320, 320))
        audit = model.parameter_ledger()
        self.assertEqual(audit['total'], 55999989)
        self.assertEqual(audit['active'], 6967989)
        self.assertEqual(audit['routed'], 16 * audit['active_routed'])

    def test_norm_does_not_depend_on_crop_extent(self):
        norm = PixelNorm(8)
        image = torch.randn(2, 8, 12, 19)
        torch.testing.assert_close(norm(image)[:, :, 2:9, 4:13], norm(image[:, :, 2:9, 4:13]))

    def test_resize_forward_and_gradient_match_bilinear_reference(self):
        for hw in ((17, 9), (3, 4), (1, 1)):
            x = torch.randn(2, 3, 7, 11, requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            actual = resize_bilinear(x, hw)
            expected = torch.nn.functional.interpolate(y, size=hw, mode='bilinear', align_corners=False)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            grad = torch.randn_like(actual)
            actual.backward(grad)
            expected.backward(grad)
            torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=2e-5)

    def test_only_selected_expert_runs_and_router_gets_gradient(self):
        ffn = Top1FFN(8, 16)
        with torch.no_grad():
            ffn.router.weight.zero_()
            ffn.router.bias.zero_()  # deterministic first expert on ties
        visits = []
        handles = [expert.register_forward_hook(lambda module, args, out, e=e: visits.append((e, len(args[0]))))
                   for e, expert in enumerate(ffn.bank)]
        image = torch.randn(1, 8, 4, 7, requires_grad=True)
        result = ffn(image)
        flat = image.permute(0, 2, 3, 1).reshape(-1, 8)
        expected = ffn.bank[0](flat) / 16
        torch.testing.assert_close(result.permute(0, 2, 3, 1).reshape(-1, 8), expected)
        result.square().mean().backward()
        for handle in handles:
            handle.remove()
        self.assertEqual(visits[:1], [(0, 28)])
        self.assertFalse(any(e for e, _ in visits))
        self.assertGreater(float(ffn.router.weight.grad.abs().sum()), 0)
        self.assertIsNone(ffn.bank[1][0].weight.grad)

    def test_rectangular_loss_backward_and_batch_equivalence(self):
        torch.manual_seed(41)
        cfg = load_config(ROOT / 'remote/run_cnn_single_56m.json')
        helpers = load_helpers()
        losses, features, alignment = helpers
        geometry = small_geometry()
        model = SingleFrameCNN(tiny_shape(), geometry)
        model.capture_enabled = True
        aligner = make_aligner(model, losses)
        chunks = []
        for index in range(2):
            image = np.random.default_rng(index).uniform(.1, .9, (48, 64, 3)).astype(np.float32)
            item = dict(proxy=image, hw=(48, 64), geometry=geometry,
                        controls=cfg['controls'], noise_seed=index)
            attach_features(item, features)
            np.testing.assert_array_equal(item['features'][:, 4:7], item['features'][:, 7:10])
            item['t_head'] = np.full((48, 64, 4), .012, dtype=np.float32)
            item['t_res'] = {}
            for tag, dims in TEACHER_DIMS.items():
                name = 'd5' if tag == 's-vit' else tag.split('-')[-1]
                rows = geometry['full_rows'] if name == 'full' else geometry['levels'][int(name[1:])]['rows']
                item['t_res'][alignment[tag]] = np.ones((rows, dims), dtype=np.float16) * .01
            chunks.append(item)
        batch_x = torch.from_numpy(np.stack([x['features'] for x in chunks])).reshape(-1, 16)
        with torch.no_grad():
            batch = model(batch_x, batch=2).reshape(2, -1, 4)
            singles = torch.stack([model(torch.from_numpy(x['features'])) for x in chunks])
        torch.testing.assert_close(batch, singles, atol=2e-6, rtol=2e-5)
        loss, metrics = chunk_loss(model, aligner, chunks, cfg, helpers)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss.detach()), float(metrics['loss']), places=6)
        loss.backward()
        self.assertGreater(float(model.stem[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(aligner.proj['s_enc_full'].weight.grad.abs().sum()), 0)
        # Unsupervised temporal logit must not receive a spatial target gradient.
        self.assertEqual(float(model.head.weight.grad[3].abs().sum()), 0)

    def test_new_package_roundtrip(self):
        from nr_package import load_student_package, write_package
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            shape = tiny_shape()
            shape_path = folder / 'shape_source.json'
            shape_path.write_text(json.dumps(shape), encoding='utf-8')
            model = SingleFrameCNN(shape, geometry_from_valid(320, 320))
            manifest = write_package(model, model.state_dict(), folder / 'package', shape_path)
            restored, _ = load_student_package(folder / 'package', 320, 320, 'cpu')
            self.assertEqual(manifest['model']['parameterAudit'], model.parameter_ledger())
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)


class SingleDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = load_config(ROOT / 'remote/run_cnn_single_56m.json')
        self.cfg.update(data_root=str(self.root), val_fraction=.25, sources=[
            dict(name='photos', directory='photos', weight=.5, group_by='file', frame_stride=1),
            dict(name='video', directory='video', weight=.5, group_by='directory', frame_stride=1)])
        rois = []
        for index, name in enumerate(['photos/a.png', 'photos/b.png', 'photos/c.png', 'photos/d.png',
                                      'video/clip_a/001.png', 'video/clip_a/002.png',
                                      'video/clip_b/001.png', 'video/clip_b/002.png']):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            h, w = (640, 800) if index % 2 else (800, 640)
            array = np.random.default_rng(index).integers(0, 256, (h, w, 3), dtype=np.uint8)
            Image.fromarray(array).save(path)
            rois.append(dict(path=name, rois=[[w // 3, h // 3, 2 * w // 3, 2 * h // 3]]))
        # The same frame links both videos; their other frames must stay together.
        (self.root / 'video/clip_b/001.png').write_bytes((self.root / 'video/clip_a/001.png').read_bytes())
        rois[-2]['rois'] = rois[-4]['rois']
        roi_path = self.root / 'rois.jsonl'
        roi_path.write_text('\n'.join(json.dumps(x) for x in rois), encoding='utf-8')
        self.manifest = build_manifest(self.cfg, roi_path=roi_path)
        self.path = self.root / 'manifest.json'
        self.path.write_text(json.dumps(self.manifest), encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def source(self, mode=None):
        cfg = copy.deepcopy(self.cfg)
        if mode:
            cfg['sampling'] = {key: float(key == mode) for key in cfg['sampling']}
        return SingleFrameSource(self.path, cfg)

    def test_split_dedup_and_video_group_isolation(self):
        self.assertEqual(self.manifest['counts']['duplicates'], 1)
        video = [x for x in self.manifest['entries'] if x['source'] == 'video']
        self.assertEqual(len({x['split_group'] for x in video}), 1)
        self.assertEqual(len({x['split'] for x in video}), 1)
        source = self.source()
        self.assertFalse({x['sha256'] for x in source.train}.intersection(x['sha256'] for x in source.val))

    def test_explicit_identity_group_joins_different_sources(self):
        path = self.root / 'groups.jsonl'
        path.write_text('\n'.join(json.dumps(dict(path=p, group='same-person')) for p in
                                   ('photos/a.png', 'video/clip_b/002.png')), encoding='utf-8')
        manifest = build_manifest(self.cfg, groups_path=path)
        chosen = [x for x in manifest['entries'] if x['group'] == 'explicit/same-person']
        self.assertEqual(len(chosen), 2)
        self.assertEqual(len({x['split_group'] for x in chosen}), 1)
        self.assertEqual(len({x['split'] for x in chosen}), 1)

    def test_native_crop_preserves_exact_input_pixels(self):
        for mode in ('roi_native', 'random_native'):
            source = self.source(mode)
            for index in range(8):
                item = source.sample(index)
                m = item['metadata']
                expected = np.asarray(read_rgb(m['path']).crop(m['crop_box']), dtype=np.float32) / 255
                np.testing.assert_array_equal(item['proxy'], expected)
                self.assertEqual(m['scale'], 1)
                self.assertEqual(m['mode'], mode)
                for box in m['valid_rois']:
                    self.assertTrue(0 <= box[0] < box[2] <= item['hw'][1])
                    self.assertTrue(0 <= box[1] < box[3] <= item['hw'][0])

    def test_context_aspect_ratio_and_real_valid_rectangle(self):
        source = self.source('context')
        for index in range(8):
            item = source.sample(index)
            h, w = item['hw']
            ow, oh = item['metadata']['original_size']
            self.assertLessEqual(abs(w - ow * h / oh), 1)
            self.assertEqual(item['geometry']['valid_width'], w)
            self.assertEqual(item['geometry']['valid_height'], h)
            self.assertLessEqual(item['geometry']['full_rows'], self.cfg['batch_rows'])

    def test_reproducible_across_worker_completion_order(self):
        source = self.source()
        indices = list(range(12))
        serial = [source.sample(i) for i in indices]
        with ThreadPoolExecutor(max_workers=4) as pool:
            parallel = list(pool.map(source.sample, reversed(indices)))
        for first, second in zip(serial, reversed(parallel)):
            self.assertEqual(first['metadata'], second['metadata'])
            np.testing.assert_array_equal(first['proxy'], second['proxy'])

    def test_roi_category_never_silently_becomes_random(self):
        for item in self.manifest['entries']:
            item['rois'] = []
        self.path.write_text(json.dumps(self.manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'no eligible training ROI'):
            self.source()

    def test_validation_keeps_native_size_and_restores_model(self):
        from train_distill_single import load_helpers
        _, build_features, _ = load_helpers()
        source = self.source()
        cfg = copy.deepcopy(self.cfg)
        cfg.update(val_count=1, validation_max_rows=2000000)
        model = SingleFrameCNN(tiny_shape(), geometry_from_valid(320, 320))
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema = {k: v.clone() for k, v in state.items()}
        ema['head.bias'] += .1  # eval must actually swap weights, then restore raw
        class FakeTeacher:
            budget = 2000000
            def prepare(self, items, capture=False):
                for item in items:
                    h, w = item['hw']
                    item['t_head'] = np.zeros((h, w, 4), dtype=np.float32)
                return items
        means = validate(model, ema, source, FakeTeacher(), cfg, build_features, 0, self.root)
        self.assertEqual(means['native']['count'], 1)
        report = json.loads((self.root / 'validation_000000.json').read_text())
        native = next(r for r in report['images'] if r['mode'] == 'native')
        self.assertEqual(native['valid_hw'], list(reversed(native['original_size'])))
        self.assertGreater(native['roi_pixels'], 0)
        self.assertTrue(np.isfinite(native['psnr_roi']))
        self.assertEqual(means['native']['roi_count'], 1)
        self.assertTrue(model.training)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], atol=0, rtol=0)

    def test_validation_oversize_is_reported_without_resizing(self):
        _, build_features, _ = load_helpers()
        source = self.source()
        cfg = copy.deepcopy(self.cfg)
        cfg.update(val_count=1, validation_max_rows=1)
        model = SingleFrameCNN(tiny_shape(), geometry_from_valid(320, 320))
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        class ForbiddenTeacher:
            def prepare(self, *args, **kwargs):
                raise AssertionError('oversize input must be skipped before teacher inference')
        means = validate(model, state, source, ForbiddenTeacher(), cfg, build_features, 1, self.root)
        self.assertTrue(all(m['count'] == 0 for m in means.values()))
        report = json.loads((self.root / 'validation_000001.json').read_text())
        self.assertTrue(all('skipped' in r for r in report['images']))


if __name__ == '__main__':
    torch.set_num_threads(8)
    unittest.main()
