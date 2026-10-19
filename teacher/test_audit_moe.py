"""Parameter accounting regression fixtures; runs with stdlib-only Python -S."""
import copy
import math
import unittest

from audit_moe import _routing, audit_manifest
from moe_shape import candidate, ledger


def tensor(name, dims, parameter=True):
    return dict(name=name, shape=dims, dtype='f32', byteLength=4 * math.prod(dims),
                isParameter=parameter)


def cnn_fixture(experts=8, packed=False):
    shape = dict(architecture=('cnn-unet-fast8-v1' if packed else
                               'cnn-single-moe16-v1' if experts == 16 else 'cnn_unet_moe8'),
                 experts=experts, channels={'full': 8}, expert_hidden={'full': 13},
                 blocks_enc={'full': 1}, blocks_dec={})
    prefix = 'enc.full.0.ffn.'
    entries = [tensor('stem.0.weight', [8, 16, 3, 3]), tensor('head.weight', [4, 8, 1, 1]),
               tensor('head.bias', [4]), tensor(prefix + 'router.weight', [experts, 8, 1, 1]),
               tensor(prefix + 'router.bias', [experts]), tensor('debug_buffer', [experts], False)]
    if packed:
        entries += [tensor(prefix + key, dims) for key, dims in
                    [('w1', [experts, 8, 13]), ('b1', [experts, 13]),
                     ('w2', [experts, 13, 8]), ('b2', [experts, 8])]]
    else:
        entries += [tensor(prefix + f'bank.{index}.' + key, dims) for index in range(experts)
                    for key, dims in [('0.weight', [13, 8]), ('0.bias', [13]),
                                      ('2.weight', [8, 13]), ('2.bias', [8])]]
    return shape, entries


class AuditMoETest(unittest.TestCase):
    def test_cnn8_counts_linear_biases_and_excludes_router(self):
        shape, entries = cnn_fixture()
        result = audit_manifest({'tensors': entries}, shape)
        totals = result['totals']
        self.assertEqual(totals['parameters'], 3092)
        self.assertEqual(totals['public_parameters'], 1260)
        self.assertEqual(totals['expert_bank_parameters'], 1832)
        self.assertEqual(totals['active_parameter_equivalent'], 1489)
        self.assertEqual(totals['buffers'], 8)
        self.assertEqual(result['public_by_category']['router'], 72)
        self.assertTrue(all(type(value) is int for value in totals.values()))

    def test_single16_keeps_fractional_individual_bias_contributions_exact(self):
        shape, entries = cnn_fixture(experts=16)
        totals = audit_manifest({'tensors': entries}, shape)['totals']
        # One expert = 8*13 + 13 + 13*8 + 8 = 229 parameters.
        self.assertEqual(totals['parameters'], 4996)
        self.assertEqual(totals['public_parameters'], 1332)
        self.assertEqual(totals['selected_routed_equivalent'], 229)
        self.assertEqual(totals['active_parameter_equivalent'], 1561)

    def test_packed_and_per_expert_layouts_have_equal_totals(self):
        old_shape, old_entries = cnn_fixture()
        new_shape, new_entries = cnn_fixture(packed=True)
        self.assertEqual(audit_manifest({'tensors': old_entries}, old_shape)['totals'],
                         audit_manifest({'tensors': new_entries}, new_shape)['totals'])

    def test_missing_expert_bias_or_entire_bank_is_rejected(self):
        shape, entries = cnn_fixture()
        for missing in ('enc.full.0.ffn.bank.7.0.bias', 'entire_bank'):
            subset = [entry for entry in entries if entry['name'] != missing and
                      not (missing == 'entire_bank' and '.bank.' in entry['name'])]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                audit_manifest({'tensors': subset}, shape)

    def test_bad_packed_bank_axis_or_hidden_size_is_rejected(self):
        shape, entries = cnn_fixture(packed=True)
        for dims in ([7, 8, 13], [8, 8, 12]):
            malformed = copy.deepcopy(entries)
            index = next(i for i, entry in enumerate(entries) if entry['name'].endswith('ffn.w1'))
            malformed[index] = tensor(malformed[index]['name'], dims)
            with self.subTest(dims=dims), self.assertRaises(ValueError):
                audit_manifest({'tensors': malformed}, shape)

    def test_unknown_or_out_of_range_expert_is_rejected(self):
        shape, entries = cnn_fixture()
        for name in ('enc.full.0.ffn.bank.8.0.bias', 'enc.full.1.ffn.bank.7.0.bias',
                     'enc.full.0.ffn.unknown'):
            malformed = copy.deepcopy(entries)
            malformed[-3]['name'] = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit_manifest({'tensors': malformed}, shape)

    def test_routed7_count_and_layout_checks_without_old_80m_target(self):
        shape = candidate([32] * 6, [32] * 6, 32, 32, blocks=(2,) * 6, vit_blocks=0)
        result = audit_manifest({'tensors': ledger(shape)}, shape)
        totals = result['totals']
        self.assertEqual(totals['expert_bank_parameters'], 540064)
        self.assertEqual(totals['selected_routed_equivalent'], 77152)
        self.assertEqual(totals['public_parameters'], 49199)
        self.assertEqual(totals['parameters'], 589263)
        self.assertEqual(totals['active_parameter_equivalent'], 126351)
        self.assertEqual(totals['buffers'], 161)
        self.assertFalse(result['budget']['applicable'])
        self.assertTrue(result['budget']['pass'])
        incomplete = [entry for entry in ledger(shape) if entry['name'] != 'vit_out.weight']
        failed = audit_manifest({'tensors': incomplete}, shape)['budget']
        self.assertFalse(failed['layout_pass'])
        self.assertFalse(failed['transitions_pass'])
        self.assertFalse(failed['pass'])

    def test_prototype_budget_remains_an_explicit_strict_check(self):
        shape = candidate([32] * 6, [32] * 6, 32, 32, blocks=(2,) * 6, vit_blocks=0)
        budget = audit_manifest({'tensors': ledger(shape)}, shape, prototype_budget=True)['budget']
        self.assertTrue(budget['applicable'])
        self.assertEqual(budget['target_parameters'], 80_000_000)
        self.assertFalse(budget['total_pass'])
        self.assertFalse(budget['pass'])

    def test_routing_axis_includes_shared_slices_and_topk_is_bounded(self):
        self.assertEqual(_routing({'routing': {'routed_experts': 16, 'shared_experts': 1, 'top_k': 2}}),
                         (17, 1, 2))
        self.assertEqual(_routing({}), (7, 0, 1))
        with self.assertRaises(ValueError):
            _routing({'routing': {'routed_experts': 7, 'top_k': 8}})

    def test_legacy_moe_tensor_accounting_is_unchanged(self):
        shape = dict(levels=[dict(level='full', kind='moe', experts=8)], vit={})
        entries = [tensor('enc_blocks.full.0.expand', [8, 2, 3]), tensor('head.weight', [4, 2])]
        totals = audit_manifest({'tensors': entries}, shape)['totals']
        self.assertEqual(totals['parameters'], 56)
        self.assertEqual(totals['active_parameter_equivalent'], 14)


if __name__ == '__main__':
    unittest.main()
