#!/usr/bin/env python3
# Checks the Python oracle in nr_numerics.py against web/fixtures/numerics.bin - the record of what the
# Vulkan reference (src/reference.cpp) produced for every case src/numerics_cases.js names. The fixture
# stores results and not inputs: both sides redraw them from the same 32-bit xorshift.
#
#   python3 check_numerics.py [path/to/numerics.bin]
#
# With --torch it also runs the vectorized torch kernels in nr_torch.py over the same cases, which is what
# pins the GPU-side arithmetic to the oracle.

import os
import struct
import sys

import nr_numerics as num


class Xorshift:
    """The generator the dumper drew its inputs from."""

    def __init__(self, seed):
        self.state = seed & 0xffffffff

    def next(self):
        x = self.state
        x = (x ^ ((x << 13) & 0xffffffff)) & 0xffffffff
        x = (x ^ (x >> 17)) & 0xffffffff
        x = (x ^ ((x << 5) & 0xffffffff)) & 0xffffffff
        self.state = x
        return x


def finite_half(draw):
    """Non-finite operands take separately listed paths in the kernels, so the random cases stay finite."""
    bits = draw & 0xffff
    if (bits & 0x7c00) == 0x7c00:
        bits &= 0x7bff
    return bits


def is_half_nan(bits):
    return (bits & 0x7c00) == 0x7c00 and (bits & 0x03ff) != 0


def read_fixture(path):
    with open(path, 'rb') as handle:
        data = handle.read()
    if data[:8] != b'NRNUM001':
        raise SystemExit(f'not a numerics fixture: {data[:8]!r}')
    count = struct.unpack_from('<I', data, 8)[0]
    sections = {}
    offset = 12
    for _ in range(count):
        tag = data[offset:offset + 4].decode('ascii')
        length = struct.unpack_from('<I', data, offset + 4)[0]
        sections[tag] = data[offset + 8:offset + 8 + length]
        offset += 8 + length
    return sections


def halves(sections, tag):
    raw = sections[tag]
    return list(struct.unpack(f'<{len(raw) // 2}H', raw))


def words(sections, tag):
    raw = sections[tag]
    return list(struct.unpack(f'<{len(raw) // 4}I', raw))


def numerics_cases(sections):
    """The cases the fixture answers: (name, count, expected, kind, skip_nonfinite, draw(case_index))."""
    cases = []

    expected = halves(sections, 'F16B')
    rng = Xorshift(0x9e3779b9)
    inputs = [rng.next() for _ in range(len(expected))]
    cases.append({
        'name': 'f16Bits over arbitrary f32 patterns',
        'count': len(expected), 'expected': expected, 'kind': 'half',
        'actual': lambda i: num.f16_bits(num.f32_from_bits(inputs[i])),
    })

    cases.append({
        'name': 'e4m3FromF16Bits over every half',
        'count': 65536, 'expected': list(sections['E4EN']), 'kind': 'byte',
        'actual': lambda i: num.e4m3_from_f16_bits(i),
    })

    cases.append({
        'name': 'e4m3ToNumber over every byte',
        'count': 256, 'expected': words(sections, 'E4DE'), 'kind': 'word',
        'actual': lambda i: num.f32_bits(num.e4m3_to_number(i)),
    })

    cases.append({
        'name': 'mpCubicSilu over every half',
        'count': 65536, 'expected': halves(sections, 'SILU'), 'kind': 'half', 'skip_nonfinite': True,
        'actual': lambda i: num.f16_bits(num.mp_cubic_silu(num.f16_to_number(i))),
    })

    cases.append({
        'name': 'expWeight over every half',
        'count': 65536, 'expected': halves(sections, 'EXPW'), 'kind': 'half', 'skip_nonfinite': True,
        'actual': lambda i: num.f16_bits(num.exp_weight(num.f16_to_number(i))),
    })

    expected = halves(sections, 'FDP8')
    rng = Xorshift(0x85ebca6b)
    fdpa_operands = []
    for _ in range(len(expected)):
        a = [num.e4m3_to_number(rng.next() & 0xff) for _ in range(16)]
        b = [num.e4m3_to_number(rng.next() & 0xff) for _ in range(16)]
        acc = num.f16_to_number(finite_half(rng.next()))
        fdpa_operands.append((a, b, acc))
    cases.append({
        'name': 'adaFp8Fdpa16 over random E4M3 operands',
        'count': len(expected), 'expected': expected, 'kind': 'half',
        'actual': lambda i: num.f16_bits(num.ada_fp8_fdpa16(fdpa_operands[i][0], fdpa_operands[i][1], 16,
                                                           fdpa_operands[i][2])),
        'operands': fdpa_operands, 'width': 16,
    })

    expected = halves(sections, 'FD16')
    rng = Xorshift(0xc2b2ae35)
    fd16_operands = []
    for _ in range(len(expected)):
        a = [num.f16_to_number(finite_half(rng.next())) for _ in range(8)]
        b = [num.f16_to_number(finite_half(rng.next())) for _ in range(8)]
        acc = num.f16_to_number(finite_half(rng.next()))
        fd16_operands.append((a, b, acc))
    cases.append({
        'name': 'adaF16Fdpa8 over random half operands',
        'count': len(expected), 'expected': expected, 'kind': 'half',
        'actual': lambda i: num.f16_bits(num.ada_f16_fdpa8(fd16_operands[i][0], fd16_operands[i][1], 8,
                                                          fd16_operands[i][2])),
        'operands': fd16_operands, 'width': 8,
    })

    return cases


def compare(entry, produced):
    """NaN half patterns compare equal to each other and are counted apart (a JS number cannot carry a NaN
    payload or sign, so the oracle publishes the canonical one where the reference keeps the input's)."""
    mismatches = 0
    nan_pairs = 0
    skipped = 0
    first = None
    for i in range(entry['count']):
        if entry.get('skip_nonfinite') and (i & 0x7c00) == 0x7c00:
            skipped += 1
            continue
        expected = entry['expected'][i]
        actual = produced(i)
        if expected == actual:
            continue
        if entry['kind'] == 'half' and is_half_nan(expected) and is_half_nan(actual):
            nan_pairs += 1
            continue
        mismatches += 1
        if first is None:
            first = (i, expected, actual)
    return mismatches, nan_pairs, skipped, first


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    want_torch = '--torch' in sys.argv
    here = os.path.dirname(os.path.abspath(__file__))
    fixture = args[0] if args else os.path.join(here, 'fixtures', 'numerics.bin')
    sections = read_fixture(fixture)
    cases = numerics_cases(sections)

    failed = False
    for entry in cases:
        mismatches, nan_pairs, skipped, first = compare(entry, entry['actual'])
        status = 'ok' if mismatches == 0 else 'FAIL'
        failed |= mismatches != 0
        line = f"{status:4} {entry['name']}: {entry['count']} cases"
        if nan_pairs:
            line += f", {nan_pairs} NaN pairs"
        if skipped:
            line += f", {skipped} skipped"
        if first is not None:
            line += f", first mismatch at {first[0]}: expected 0x{first[1]:04x} got 0x{first[2]:04x}"
        print(line)

    if want_torch:
        import nr_torch as nrt
        failed |= nrt.check_against_oracle(cases)

    print('CHECK NUMERICS PASS' if not failed else 'CHECK NUMERICS FAIL')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
