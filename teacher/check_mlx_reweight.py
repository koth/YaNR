#!/usr/bin/env python3
# Settle the weight-layout question with their own model: re-order their decoded logical weights back
# into our k ordering (the two decodes differ by a fixed within-tile row permutation - compute it from
# the two index formulas) and run their reference model on the same features. If the head then matches
# our runtime's head, their model is fine and the disagreement lives entirely in their decode step -
# i.e. our unpacking of the DLL is the right one.
#
#   python3 check_mlx_reweight.py <logical.safetensors> <features.bin> <our_head.bin> <reweighted-head.bin>

import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tools', 'MLX-DLSS', 'python'))

QMMA_ROW_SOURCE_16 = np.array([0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15], dtype=np.int64)


def nr_index(k, n, out_channels):
    k_tile, k_in = k >> 5, k & 31
    n_in = n & 127
    lane = (((n_in & 15) & 7) << 2) | ((k_in & 15) >> 2)
    byte = ((((n_in & 15) >> 3) << 3) | ((k_in >> 4) << 2) | (k_in & 3))
    return (k_tile * out_channels * 32 + (n >> 7) * 4096 + (n_in >> 6) * 2048
            + ((n_in & 63) >> 4) * 512 + lane * 16 + byte)


def mlx_index(k, n, out_channels):
    k_in = ((k & ~15) + QMMA_ROW_SOURCE_16[k & 15]) & 31
    n_in = n & 31
    lane = ((k_in >> 2) & 3) | ((n_in & 7) << 2)
    element = (k_in & 3) | (((k_in >> 4) & 1) << 2)
    return (k >> 5) * out_channels * 32 + (n >> 5) * 1024 \
        + ((n_in >> 4) & 1) * 512 + lane * 16 + ((n_in >> 3) & 1) * 8 + element


def row_map():
    """their row -> our row: the k permutation shared by every packed matrix."""
    out_channels = 128
    their_to_ours = {}
    for kp in range(32):
        theirs = [mlx_index(kp, n, out_channels) for n in range(32)]
        for kn in range(32):
            if [nr_index(kn, n, out_channels) for n in range(32)] == theirs:
                their_to_ours[kp] = kn
                break
    return their_to_ours


def main():
    packed_path, features_path, our_head_path, out_path = sys.argv[1:5]
    import pathlib
    import torch
    import mlxdlss.tools.unpack_dlssnr_weights as unpack
    from mlxdlss.pipeline import NeuralRenderingPipeline, load_weights

    mapping = row_map()
    perm = np.array([mapping[i] for i in range(32)], dtype=np.int64)   # perm[their row] = our row
    print('row permutation (their k -> our k):', perm.tolist())

    # Re-order every decoded E4M3 matrix into our k ordering at the source: their unpacker is the one
    # that knows all the record families (experts, splits, ViT); the permutation is the same 32-row map.
    original = unpack.unpack_qmma_e4m3_matrix

    def reordered(packed, *, input_features, output_features, tile_order='k-major'):
        logical = original(packed, input_features=input_features,
                           output_features=output_features, tile_order=tile_order)
        rows = np.arange(input_features, dtype=np.int64)
        source = (rows // 32) * 32 + perm[rows % 32]        # their row -> our row, per 32-row tile
        out = np.empty_like(logical)
        out[source] = logical
        return out

    unpack.unpack_qmma_e4m3_matrix = reordered
    reweighted = os.path.splitext(out_path)[0] + '.safetensors'
    saved_argv = sys.argv
    sys.argv = ['unpack_dlssnr_weights', packed_path, reweighted]
    try:
        unpack.main()
    finally:
        sys.argv = saved_argv
    weights = load_weights(reweighted)
    print(f're-decoded {len(weights)} tensors into our k ordering -> {reweighted}')

    pipeline = NeuralRenderingPipeline(weights, device='cpu', precision='reference')
    features = np.fromfile(features_path, dtype=np.float32, offset=24)
    with open(features_path, 'rb') as handle:
        dims = np.fromfile(handle, dtype=np.int64, count=3)
    head = pipeline.run_features(features.reshape(tuple(int(d) for d in dims)))
    with open(out_path, 'wb') as handle:
        np.asarray(head.shape, dtype=np.int64).tofile(handle)
        head.astype(np.float32).tofile(handle)

    ours = np.fromfile(our_head_path, dtype=np.float32, offset=24).reshape(head.shape)
    d = np.abs(head.astype(np.float64) - ours.astype(np.float64))
    print(f'reweighted-mlx head vs our head: MAE {d.mean():.6f}  max {d.max():.6f}  '
          f'per-channel {np.round(d.reshape(-1, 4).mean(0), 6).tolist()}')
    for c, name in enumerate('rgba'):
        a = head[..., c].ravel().astype(np.float64)
        b = ours[..., c].ravel().astype(np.float64)
        print(f'  ch{c} {name}: corr {np.corrcoef(a, b)[0, 1]:+.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
