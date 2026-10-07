#!/usr/bin/env python3
"""Unit tests for the cost model (openspec tasks 1.2 / 1.7).

    python3 test_cost_model.py
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cost_model import (ATTN_MACS_PER_CH, account, block_macs_per_px, load_shape,          # noqa: E402
                        teacher_shape, vit_macs_per_token)
from nr_geometry import geometry_from_valid                                                 # noqa: E402

SHAPES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shapes')


class Formulas(unittest.TestCase):
    """分项公式逐项对账(手算值)。"""

    def test_plain_block(self):
        level = {'level': 'd0', 'kind': 'ffn', 'channels': 32, 'hidden': 128,
                 'blocks': 1, 'attn': True}
        # 2*32*128 + 4*32^2 + 128*32 = 8192 + 4096 + 4096
        self.assertEqual(block_macs_per_px(level), 16384.0)

    def test_ffn_only_block(self):
        level = {'level': 'full', 'kind': 'ffn', 'channels': 32, 'hidden': 32,
                 'blocks': 1, 'attn': False}
        # 2*32*32 + 4*32^2 = 2048 + 4096
        self.assertEqual(block_macs_per_px(level), 6144.0)

    def test_expert_block(self):
        level = {'level': 'd3', 'kind': 'expert', 'channels': 256, 'hidden': 128,
                 'blocks': 1, 'attn': True}
        # ch^2*(h/32+1) + ch*h + 4ch^2 + 128ch = 360448 + 262144 + 32768
        self.assertEqual(block_macs_per_px(level), 655360.0)

    def test_split_block(self):
        level = {'level': 'd4', 'kind': 'split', 'channels': 512, 'hidden': 128,
                 'blocks': 1, 'attn': True, 'branches': 8,
                 'branch_channels': 64, 'middle_channels': 256}
        # 6ch^2 + 2*E*bc*mc + 128ch = 1572864 + 262144 + 65536
        self.assertEqual(block_macs_per_px(level), 1900544.0)

    def test_vit_block(self):
        level = {'channels': 1024, 'ffn': 4096, 'blocks': 1}
        # 2*ch*ffn + 4ch^2 = 8388608 + 4194304
        self.assertEqual(vit_macs_per_token(level), 12582912.0)

    def test_attention_constant(self):
        # 8x8 窗口 64 槽 qk+av,heads = ch/32 -> 4096/px/head -> 128*ch
        self.assertEqual(ATTN_MACS_PER_CH, 128.0)


class TeacherReplay(unittest.TestCase):
    """任务 1.3:teacher 形状重放 68.5 G ±3%(实测锚定)。"""

    def test_total_512(self):
        g = geometry_from_valid(512, 512)
        total = sum(r[4] for r in account(teacher_shape(), g))
        self.assertAlmostEqual(total, 68.5, delta=68.5 * 0.03)

    def test_stage_shares_flat(self):
        g = geometry_from_valid(512, 512)
        rows = [r for r in account(teacher_shape(), g) if r[0] not in ('transition',)]
        total = sum(r[4] for r in rows)
        for stage, _, _, _, gm in rows:                        # 教师成本在各级是平的(10-20%)
            self.assertGreater(gm / total, 0.09, stage)
            self.assertLess(gm / total, 0.22, stage)

    def test_monotonic_in_width(self):
        g = geometry_from_valid(512, 512)
        wide = teacher_shape()
        wide['levels'][2]['channels'] = 128                    # d1: 64 -> 128
        total_a = sum(r[4] for r in account(teacher_shape(), g))
        total_b = sum(r[4] for r in account(wide, g))
        self.assertGreater(total_b, total_a)


class ShapeDsl(unittest.TestCase):
    """任务 1.5:非法形状报错;v0 形状过预测门禁。"""

    def _write(self, spec):
        fd, path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as fh:
            json.dump(spec, fh)
        self.addCleanup(os.unlink, path)
        return path

    def test_rejects_bad_channels(self):
        spec = {'levels': [{'level': 'd0', 'kind': 'ffn', 'channels': 48, 'hidden': 32,
                            'blocks': 1, 'attn': False}], 'vit': {'channels': 512, 'ffn': 1024}}
        with self.assertRaises(ValueError):
            load_shape(self._write(spec))

    def test_rejects_split_without_branches(self):
        spec = {'levels': [{'level': 'd4', 'kind': 'split', 'channels': 256, 'hidden': 64,
                            'blocks': 1, 'attn': True}], 'vit': {'channels': 512, 'ffn': 1024}}
        with self.assertRaises(ValueError):
            load_shape(self._write(spec))

    def test_student_v0_gate(self):
        path = os.path.join(SHAPES_DIR, 'student_v0.json')
        student = load_shape(path)
        g = geometry_from_valid(512, 512)
        student_total = sum(r[4] for r in account(student, g))
        teacher_total = sum(r[4] for r in account(teacher_shape(), g))
        self.assertGreaterEqual(teacher_total / student_total, 5.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
