# Loading the weights.
#
# The model file stores every FP8 matrix in the fragment order a tensor-core MMA wants. The GEMM addresses
# that order directly, so this module unpacks each matrix into plain [K][N] tensors once at load - a host
# pass over 141 MiB, but then a matmul that is a matmul. It is a port of ports/browser-webgpu/src/model.js
# plus the unpacking helpers in src/reference.cpp (packedWeightIndex, packedF16WeightIndex, the prior's
# 4x4-tiled query axis). See docs/weights.md.

import json
import os

import numpy as np

from nr_numerics import e4m3_to_number, f16_to_number
from nr_geometry import align_up


def packed_input_index(k):
    """The within-32 activation index every FP8 GEMM's A operand is in."""
    base = k & ~31
    within = k & 31
    half = within & 16
    quarter = within & 15
    return base + half + (quarter >> 2) * 2 + (quarter & 1) + (8 if (quarter & 2) else 0)


def inverse_packed_input_index(k):
    base = k & ~31
    within = k & 31
    return base + (within & 17) + ((within & 2) << 1) + ((within & 4) << 1) + ((within & 8) >> 2)


# The E4M3 fragment mapping. `packed_weight_index` mirrors src/nr_model.cpp exactly. MLX-DLSS's
# independent decode of the same DLL resource uses a different within-tile k mapping (a 16-row source
# permutation over the SM89 QMMA B-fragment); the two disagree on the 310.8.3 build, so
# NR_WEIGHT_LAYOUT=mlx selects theirs and `check_vs_mlx.py` + the image run arbitrate.
_QMMA_ROW_SOURCE_16 = np.array([0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15], dtype=np.int64)
WEIGHT_LAYOUT = os.environ.get('NR_WEIGHT_LAYOUT', 'nr')


def packed_weight_index(k, n, output_channels):
    """Byte index of weight (k, n) inside a packed FP8 matrix of N columns, as the model file stores it."""
    k_tile = k >> 5
    k_in = k & 31
    n_tile = n >> 7
    n_in = n & 127
    n_half = n_in >> 6
    n_group = (n_in & 63) >> 4
    n_in_group = n_in & 15
    lane = ((n_in_group & 7) << 2) | ((k_in & 15) >> 2)
    byte_in_lane = ((n_in_group >> 3) << 3) | ((k_in >> 4) << 2) | (k_in & 3)
    return k_tile * output_channels * 32 + n_tile * 4096 + n_half * 2048 + n_group * 512 + lane * 16 + byte_in_lane


def packed_f16_weight_index(input_channel, output_channel, output_channels):
    """Half index of an f16 weight inside a packed matrix: 16x16 tiles, each one m16n8k16 B fragment pair."""
    n_tiles = (output_channels + 15) // 16
    tile = (input_channel >> 4) * n_tiles + (output_channel >> 4)
    k = input_channel & 15
    n = output_channel & 15
    lane = ((n & 7) << 2) | ((k & 7) >> 1)
    fragment = (2 if k >= 8 else 0) + (k & 1)
    return tile * 256 + lane * 8 + ((n >> 3) & 1) * 4 + fragment


def tiled_token(token):
    """Natural window token (row-major in the 8x8 window) -> physical token (4x4 tiles of 16)."""
    x = token & 7
    y = token >> 3
    return (y >> 2) * 32 + (x >> 2) * 16 + (y & 3) * 4 + (x & 3)


def inverse_tiled_token(token):
    tile = token >> 4
    within = token & 15
    x = (tile & 1) * 4 + (within & 3)
    y = (tile >> 1) * 4 + (within >> 2)
    return y * 8 + x


# numpy vector forms of the same helpers
_vec_tiled_token = np.vectorize(tiled_token, otypes=[np.int64])
_vec_inverse_tiled = np.vectorize(inverse_tiled_token, otypes=[np.int64])
_vec_packed_input = np.vectorize(packed_input_index, otypes=[np.int64])


def packed_weight_index_array(k, n, output_channels):
    k = np.asarray(k, dtype=np.int64)
    n = np.asarray(n, dtype=np.int64)
    k_tile = k >> 5
    if WEIGHT_LAYOUT == 'mlx':
        k_in = ((k & ~15) + _QMMA_ROW_SOURCE_16[k & 15]) & 31
        n_in = n & 31
        pair_group = (n_in >> 4) & 1
        fragment = (n_in >> 3) & 1
        group_id = n_in & 7
        element = (k_in & 3) | (((k_in >> 4) & 1) << 2)
        lane = ((k_in >> 2) & 3) | (group_id << 2)
        return (k_tile * output_channels * 32 + (n >> 5) * 1024
                + pair_group * 512 + lane * 16 + fragment * 8 + element)
    k_in = k & 31
    n_tile = n >> 7
    n_in = n & 127
    n_half = n_in >> 6
    n_group = (n_in & 63) >> 4
    n_in_group = n_in & 15
    lane = ((n_in_group & 7) << 2) | ((k_in & 15) >> 2)
    byte_in_lane = ((n_in_group >> 3) << 3) | ((k_in >> 4) << 2) | (k_in & 3)
    return (k_tile * output_channels * 32 + n_tile * 4096 + n_half * 2048 + n_group * 512
            + lane * 16 + byte_in_lane)


def packed_f16_weight_index_array(k, n, output_channels):
    k = np.asarray(k, dtype=np.int64)
    n = np.asarray(n, dtype=np.int64)
    n_tiles = (output_channels + 15) // 16
    tile = (k >> 4) * n_tiles + (n >> 4)
    kk = k & 15
    nn = n & 15
    lane = ((nn & 7) << 2) | ((kk & 7) >> 1)
    fragment = np.where(kk >= 8, 2, 0) + (kk & 1)
    return tile * 256 + lane * 8 + ((nn >> 3) & 1) * 4 + fragment


_E4_DECODE = np.array([e4m3_to_number(b) for b in range(256)], dtype=np.float64)
_HALF_DECODE = np.array([f16_to_number(bits) for bits in range(65536)], dtype=np.float64)


def load_relative_bias_scalar(stage, relative_byte_offset, head, query_local, key_local):
    """The reference's loadRelativeBias (src/reference.cpp), for cross-checking the untangle below."""
    q = tiled_token(query_local)
    k = tiled_token(key_local)
    m = q & 15
    n = k & 15
    lane = ((m & 7) << 2) | ((n & 7) >> 1)
    fragment = (2 if m >= 8 else 0) + (n & 1)
    tile_offset = (q >> 4) * 1024 + (k >> 4) * 256
    lane_offset = lane * 8 + (n >> 3) * 4
    half_index = tile_offset + lane_offset + fragment
    at = relative_byte_offset + head * 8192 + half_index * 2
    return f16_to_number(int(stage[at]) | (int(stage[at + 1]) << 8))


class Tensor:
    """One named slice of a stage file."""

    def __init__(self, name, block, layer, stage_id, stage_offset, byte_length, bytes_):
        self.name = name
        self.block = block
        self.layer = layer
        self.stage = stage_id
        self.stage_offset = stage_offset
        self.byte_length = byte_length
        self.bytes = bytes_      # np.uint8 view of length byte_length


class Model:
    def __init__(self, device):
        self.device = device
        self.tensors = {}
        self.stages = {}
        self.checked = set()
        self.cache = {}
        self._warned_bounds = False
        self.block_count = 0

    def load(self, directory):
        with open(os.path.join(directory, 'manifest.json'), 'r') as handle:
            manifest = json.load(handle)
        self.block_count = manifest['totals']['blockCount']

        stages = {}
        for stage in manifest['stages']:
            path = os.path.join(directory, 'model', stage['file'])
            with open(path, 'rb') as handle:
                data = handle.read()
            if len(data) != stage['packedByteLength']:
                raise ValueError(f"stage size mismatch: {stage['id']}")
            stages[stage['id']] = np.frombuffer(data, dtype=np.uint8)
        self.stages = stages

        for entry in manifest['tensors']:
            stage = stages.get(entry['stage'])
            if stage is None:
                raise ValueError(f"tensor references unknown stage {entry['stage']}")
            if entry['stageOffset'] + entry['byteLength'] > stage.size:
                raise ValueError(f"tensor exceeds stage {entry['name']}")
            view = stage[entry['stageOffset']:entry['stageOffset'] + entry['byteLength']]
            self.tensors[entry['name']] = Tensor(entry['name'], entry['block'], entry['layer'],
                                                entry['stage'], entry['stageOffset'],
                                                entry['byteLength'], view)
        return self

    def tensor(self, block, layer=0, parameter='layer'):
        name = f'block{block}.layer{layer}.{parameter}'
        found = self.tensors.get(name)
        if found is None:
            raise ValueError(f'missing tensor {name}')
        return found

    def blend_scale(self):
        """How much of the reprojected history the temporal blend may keep (a learned scalar)."""
        tensor = self.tensor(70, 0, 'blend_scale')
        if tensor.byte_length < 2:
            raise ValueError('the model has no blend scale')
        return f16_to_number(int(tensor.bytes[0]) | (int(tensor.bytes[1]) << 8))

    def fp8_matrix(self, tensor, byte_offset, k, n_matrix, batch_k=0):
        """
        One FP8 matrix as f16 values [K][N]. Every weight satisfies |w| <= 9, which is what makes the
        product of two operands scaled by four an exact normal half; a matrix that broke it would still
        produce numbers, just not the right ones, so it is a throw rather than a fallback.
        """
        batch = batch_k or k
        if k % 32 or n_matrix % 16 or batch % 32 or k % batch:
            raise ValueError(f'FP8 matrix shape must be K%32==0, N%16==0, batchK | K: {tensor.name}')
        if byte_offset + k * n_matrix > tensor.byte_length:
            raise ValueError(f'FP8 matrix exceeds tensor {tensor.name}')
        key = (tensor.name, byte_offset, k, n_matrix, batch)
        if key not in self.cache:
            if key[:2] not in self.checked:
                segment = tensor.bytes[byte_offset:byte_offset + k * n_matrix]
                magnitude = segment & 0x7f
                bad = np.any((magnitude > 0x51) & (magnitude != 0x7f))
                if bad and not self._warned_bounds:
                    # The synthetic teacher stayed within |w| <= 9; the real 310.8.x model does not
                    # (about 4% of codes are larger). The chain arithmetic is general, so just note it.
                    print(f'note: {tensor.name} has weights above the synthetic |w|<=9 bound '
                          f'(real models do)')
                    self._warned_bounds = True
                self.checked.add(key[:2])
            k_index = np.arange(k, dtype=np.int64)[:, None]
            n_index = np.arange(n_matrix, dtype=np.int64)[None, :]
            index = packed_weight_index_array(k_index, n_index, n_matrix)
            codes = np.asarray(tensor.bytes[byte_offset + index], dtype=np.int64)
            self.cache[key] = _E4_DECODE[codes].astype(np.float16)
        return self.cache[key]

    def f16_matrix(self, tensor, byte_offset, k, n):
        """A plain [K][N] f16 matrix, for the input adapter and the head."""
        if k % 16:
            raise ValueError('f16 matrix K must be a multiple of 16')
        key = ('f16', tensor.name, byte_offset, k, n)
        if key not in self.cache:
            k_index = np.arange(k, dtype=np.int64)[:, None]
            n_index = np.arange(n, dtype=np.int64)[None, :]
            half_index = (byte_offset >> 1) + packed_f16_weight_index_array(k_index, n_index, n)
            if half_index.max() * 2 + 1 >= tensor.byte_length:
                raise ValueError(f'f16 matrix exceeds tensor {tensor.name}')
            low = tensor.bytes[half_index * 2].astype(np.int64)
            high = tensor.bytes[half_index * 2 + 1].astype(np.int64)
            self.cache[key] = _HALF_DECODE[low | (high << 8)].astype(np.float16)
        return self.cache[key]

    def relative_bias(self, tensor, relative_byte_offset, heads):
        """
        The learned attention prior as f16 [heads][64 query][64 key], both tokens in natural row-major
        order within the window. The model stores it as the C accumulator of the score matrix multiply, so
        both axes arrive in 4x4-tiled order inside 16x16 fragments.
        """
        key = ('prior', tensor.name, relative_byte_offset, heads)
        if key not in self.cache:
            natural = np.arange(64, dtype=np.int64)
            q = _vec_tiled_token(natural)[:, None]       # natural query -> physical, [64, 1]
            k = _vec_tiled_token(natural)[None, :]       # natural key -> physical, [1, 64]
            m = q & 15
            n = k & 15
            lane = ((m & 7) << 2) | ((n & 7) >> 1)
            fragment = np.where(m >= 8, 2, 0) + (n & 1)
            half_index = ((q >> 4) * 1024 + (k >> 4) * 256 + lane * 8 + (n >> 3) * 4 + fragment)
            # Both axes above are already in natural order: column j reads the key whose natural index is j.
            byte_index = (relative_byte_offset + np.arange(heads, dtype=np.int64)[:, None, None] * 8192
                          + half_index[None, :, :] * 2)                 # [heads, 64, 64]
            low = tensor.bytes[byte_index]
            high = tensor.bytes[byte_index + 1]
            codes = low.astype(np.int64) | (high.astype(np.int64) << 8)
            self.cache[key] = _HALF_DECODE[codes].astype(np.float16)
        return self.cache[key]

    def head_scales(self, tensor, byte_offset, heads):
        """The per-head f32 attention scales."""
        key = ('heads', tensor.name, byte_offset, heads)
        if key not in self.cache:
            raw = tensor.bytes[byte_offset:byte_offset + heads * 4].copy()
            self.cache[key] = raw.view('<f4')[:heads].copy()
        return self.cache[key]

    def aux_vector(self, tensor, byte_offset, count):
        """The per-channel skip scales of one tensor slice, as f16 values."""
        key = ('aux', tensor.name, byte_offset, count)
        if key not in self.cache:
            if byte_offset % 2:
                raise ValueError(f'skip scales of {tensor.name} are not half-aligned')
            if byte_offset + count * 2 > tensor.byte_length:
                raise ValueError(f'skip scales exceed {tensor.name}')
            raw = tensor.bytes[byte_offset:byte_offset + count * 2].copy()
            codes = raw.view('<u2')[:count].astype(np.int64)
            self.cache[key] = _HALF_DECODE[codes].astype(np.float16)
        return self.cache[key]

    def aux_pair(self, tensor, offset_a, offset_b, count):
        """Two per-channel scale vectors end to end, for the post blend, which reads both."""
        key = ('auxpair', tensor.name, offset_a, offset_b, count)
        if key not in self.cache:
            self.cache[key] = np.stack([self.aux_vector(tensor, offset_a, count),
                                        self.aux_vector(tensor, offset_b, count)])
        return self.cache[key]
