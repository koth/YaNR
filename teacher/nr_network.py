# The network: 71 blocks over a six-level encoder/decoder with a global ViT at the bottom. A port of
# ports/browser-webgpu/src/graph.js (which mirrors src/nr_graph.cpp), with the elementwise kernels of
# shaders/ops.wgsl, the ViT of shaders/vit.wgsl and the window attention of src/reference.cpp
# (windowNormalizeRef / windowAttendRef) spelled out in torch.
#
# Tensor representation: f16 [rows, channels] torch tensors holding buffer values. E4M3 tensors hold f16
# values on the E4M3 grid; `fp8_quant` is the publication. The GEMMs are the F13/F24 fixed-point chains of
# nr_torch.py - the residual is the accumulator seed scaled by the block's learned per-channel vector, not
# an add after the fact.

import numpy as np
import torch

import nr_torch as nrt
from nr_torch import (DEVICE, fp8_quant, round_f16_t, mp_cubic_silu_t, exp_weight_t, vit_exp_weight_t,
                      fp8_dot_chain, fp8_dot_chain_batched, f16_dot_chain)

try:
    import nr_triton as ntr_unused  # noqa: F401  (imported for its tables when nr_cuda drives them)
except Exception:
    ntr_unused = None

# The fused kernels: the native CUDA module if it builds, else Triton, else the torch loops.
ntr = nrt.fused_backend()
from nr_model import packed_input_index, inverse_tiled_token
from nr_geometry import (fused_layout, pre_fused_layout, upsample_fused_layout, post_fused_layout,
                         window_phase, WindowPhases)


class Network:
    def __init__(self, model, geometry):
        self.model = model
        self.g = geometry
        self.phases = None
        self._torch = {}
        self.boundaries = None

    def capture(self, name, tensor):
        """Keep a copy of one block output, for comparing against the WebGPU port (graph.js capture)."""
        if self.boundaries is not None:
            self.boundaries[name] = tensor

    # ------------------------------------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------------------------------------

    def np16(self, arr):
        """A model numpy array as an f16 tensor on the device, converted once."""
        key = ('f16', id(arr))
        found = self._torch.get(key)
        if found is None:
            found = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float16)).to(DEVICE)
            self._torch[key] = found
        return found

    def np32(self, arr):
        key = ('f32', id(arr))
        found = self._torch.get(key)
        if found is None:
            found = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(DEVICE)
            self._torch[key] = found
        return found

    def zeros(self, rows, channels):
        return torch.zeros((rows, channels), dtype=torch.float16, device=DEVICE)

    def head_scale_const(self):
        """round_f16(sqrt(32)) as a device scalar, built once: stream capture forbids host transfers."""
        key = 'sqrt32'
        found = self._torch.get(key)
        if found is None:
            found = round_f16_t(torch.tensor(32.0, device=DEVICE).sqrt())
            self._torch[key] = found
        return found

    def _swizzled(self, x, count):
        """The A operand's within-32 index rotation, which the kernel applies at load."""
        perm = [packed_input_index(k) for k in range(count)]
        return x[:, torch.tensor(perm, dtype=torch.long, device=DEVICE)]

    # ------------------------------------------------------------------------------------------------
    # The matrix multiplies.
    # ------------------------------------------------------------------------------------------------

    def gemm(self, x, w, k, n, batches=1, broadcast=False, partition=0, silu=False,
             residual=None, aux=None, e4=True, half=False):
        """
        One FP8 matrix multiply. `residual`, when given, is not added afterwards - it is the value the
        accumulator starts from, scaled by the block's learned per-channel vector (`aux`).
        Returns (e4 tensor or None, f16 tensor or None).
        """
        rows = x.shape[0]
        batch_k = k
        count = batch_k if broadcast else batches * batch_k
        wt = self.np16(w)

        out_e4 = torch.empty((rows, batches * n), dtype=torch.float16, device=DEVICE) if e4 else None
        out_f16 = torch.empty((rows, batches * n), dtype=torch.float16, device=DEVICE) if half else None

        if ntr is not None:
            # One launch: the chain, the A-operand swizzle, the residual seed and the epilogue all inside.
            if broadcast or batches == 1:
                xv = x[:, :batch_k].unsqueeze(0).expand(batches, rows, batch_k)
            else:
                xv = x[:, :count].view(rows, batches, batch_k).permute(1, 0, 2)
            wv = wt.view(batches, batch_k, n) if batches > 1 else wt.unsqueeze(0)
            resv = residual.view(rows, batches, n).permute(1, 0, 2) if residual is not None else None
            e4v = out_e4.view(rows, batches, n).permute(1, 0, 2) if out_e4 is not None else None
            rawv = out_f16.view(rows, batches, n).permute(1, 0, 2) if out_f16 is not None else None
            if partition:
                ntr.fp8_gemm_partitioned(
                    xv, wv, partition, residual=resv, aux=aux, silu=silu, quantize=e4, raw=half,
                    out_e4=e4v, out_raw=rawv)
                return out_e4, out_f16
            else:
                # Batch strides are the per-batch offsets: an expanded (stride 0) input is the broadcast
                # case and is fine as it is.
                ntr.fp8_gemm_full(xv, wv.contiguous(),
                                  residual=resv, aux=aux, silu=silu, quantize=e4, raw=half,
                                  out_e4=e4v, out_raw=rawv)
            return out_e4, out_f16

        xs = self._swizzled(x, count)
        for b in range(batches):
            xin = xs[:, :batch_k] if (broadcast or batches == 1) else xs[:, b * batch_k:(b + 1) * batch_k]
            wb = wt if batches == 1 else wt[b * batch_k:(b + 1) * batch_k, :]
            seed = None
            if residual is not None:
                res = residual if batches == 1 else residual[:, b * n:(b + 1) * n]
                seed = round_f16_t(res.float() * aux.view(1, -1).float())
            value = fp8_dot_chain(xin, wb, seed, partition=partition)
            lo, hi = b * n, (b + 1) * n
            if half:
                out_f16[:, lo:hi] = value
            if e4:
                out_e4[:, lo:hi] = fp8_quant(mp_cubic_silu_t(value) if silu else value)
        return out_e4, out_f16

    def gemm_f16(self, x, w, k, n, e4=False, half=True, f32=False):
        """The f16 matrix multiply: only the input adapter and the head."""
        if ntr is not None:
            out_e4, out_raw = ntr.f16_gemm(x, self.np16(w), quantize=e4, raw=True)
        else:
            value = f16_dot_chain(x, self.np16(w))
            out_raw = value
            out_e4 = fp8_quant(value) if e4 else None
        return out_e4, (out_raw if half else None), (out_raw.float() if f32 else None)

    # ------------------------------------------------------------------------------------------------
    # The elementwise kernels (shaders/ops.wgsl).
    # ------------------------------------------------------------------------------------------------

    def downsample(self, in_f16, channels, in_w, in_h, out_w, out_h):
        """The 2x2 box pool: three half adds as (a+b)+(c+d), one half multiply by 0.25, E4M3 publication."""
        x = in_f16.view(in_h, in_w, channels)
        sx = (torch.arange(out_w, device=DEVICE) * 2).clamp(max=max(in_w - 1, 0))
        sy = (torch.arange(out_h, device=DEVICE) * 2).clamp(max=max(in_h - 1, 0))
        sx1 = (sx + 1).clamp(max=in_w - 1)
        sy1 = (sy + 1).clamp(max=in_h - 1)
        valid = ((sx + 1 < in_w).view(1, -1)) & ((sy + 1 < in_h).view(-1, 1))

        top = x[sy][:, sx] + x[sy][:, sx1]
        bottom = x[sy1][:, sx] + x[sy1][:, sx1]
        values = ((top + bottom) * 0.25).to(torch.float16)
        values = torch.where(valid.unsqueeze(-1), values, torch.zeros_like(values))
        return fp8_quant(values.reshape(-1, channels))

    def upsample_residual(self, in_f16, skip_e4, aux, channels, in_w, in_h, out_w, out_h, dual=False):
        """The decoder entry: the level below, doubled, plus the encoder skip times a learned scale. The
        residual product is not published before the add - the whole expression rounds once."""
        base = in_f16.view(in_h, in_w, channels)
        src_y = (torch.arange(out_h, device=DEVICE) >> 1).clamp(max=in_h - 1)
        src_x = (torch.arange(out_w, device=DEVICE) >> 1).clamp(max=in_w - 1)
        low = base[src_y][:, src_x]
        skip = skip_e4.view(out_h, out_w, channels)
        values = (low.float() + skip.float() * aux.view(1, 1, -1).float()).to(torch.float16)
        flat = values.reshape(-1, channels)
        return fp8_quant(flat), (flat if dual else None)

    def post_blend(self, in_e4, skip_e4, aux_pair, channels, in_w, in_h, out_w, out_h):
        """The last merge before block 70: the level-0 decoder output doubled and scaled, plus block 0's
        own output scaled. The first product is published to half, the second is folded into the rounding."""
        base = in_e4.view(in_h, in_w, channels)
        src_y = (torch.arange(out_h, device=DEVICE) >> 1).clamp(max=in_h - 1)
        src_x = (torch.arange(out_w, device=DEVICE) >> 1).clamp(max=in_w - 1)
        low = base[src_y][:, src_x]
        skip = skip_e4.view(out_h, out_w, channels)
        upsampled = round_f16_t(low.float() * aux_pair[0].view(1, 1, -1).float())
        values = (upsampled.float() + skip.float() * aux_pair[1].view(1, 1, -1).float()).to(torch.float16)
        flat = values.reshape(-1, channels)
        return fp8_quant(flat), flat

    # ------------------------------------------------------------------------------------------------
    # Cosine normalization and attention.
    # ------------------------------------------------------------------------------------------------

    @staticmethod
    def _norm_tree(r):
        """16 pair squares -> one total; the association both the window and the ViT kernels use."""
        t = r[..., 0:8] + r[..., 8:16]
        u = t[..., 0:4] + t[..., 4:8]
        v = u[..., 0:2] + u[..., 2:4]
        return v[..., 0:1] + v[..., 1:2]

    def _cosine_norm(self, x):
        """Unit length per head: pair squares, half tree, 1/sqrt in f32, one rounding to half."""
        low = x[..., :16]
        high = x[..., 16:]
        high_square = round_f16_t(high * high)
        r = (low.float() * low.float() + high_square.float()).to(torch.float16)
        total = self._norm_tree(r)
        return round_f16_t(1.0 / total.float().sqrt())

    @staticmethod
    def _softmax_tree(scores):
        """The 64-wide softmax denominator: pair sums over the tiled key order, half adds throughout."""
        b01 = scores[..., 0:8] + scores[..., 8:16]
        b23 = scores[..., 16:24] + scores[..., 24:32]
        b45 = scores[..., 32:40] + scores[..., 40:48]
        b67 = scores[..., 48:56] + scores[..., 56:64]
        pairs = ((b01 + b23) + b45) + b67            # index = pairIndex * 2 + parity
        even = ((pairs[..., 0:1] + pairs[..., 2:3]) + pairs[..., 4:5]) + pairs[..., 6:7]
        odd = ((pairs[..., 1:2] + pairs[..., 3:4]) + pairs[..., 5:6]) + pairs[..., 7:8]
        return (even + odd).squeeze(-1)

    def window_attention(self, qkv, prior, scales, width, height, heads, phase):
        """One shifted-window attention; the cosine normalization is part of it, off the raw qkv."""
        rows = width * height
        shift_x, shift_y = window_phase(phase)
        windows_x = (width + shift_x + 7) // 8
        windows_y = (height + shift_y + 7) // 8
        prior = self.np16(prior)

        if ntr is not None and getattr(ntr, 'FUSED_PREP', False):
            # One launch from the raw qkv: the cosine norm, the q/k/v publications, scores, expWeight,
            # the softmax tree, the value fold and the E4M3 publication are all inside the kernel.
            scale_half = round_f16_t(self.np32(scales))
            out = ntr.window_attention_raw(qkv.view(rows, heads, 96), prior, scale_half,
                                           width, height, heads, shift_x, shift_y)
            return out.reshape(rows, heads * 32)

        qkv3 = qkv.view(rows, heads, 96)
        q_norm = self._cosine_norm(qkv3[:, :, :32])
        k_norm = self._cosine_norm(qkv3[:, :, 32:64])
        scale_half = round_f16_t(self.np32(scales))

        q_pub = fp8_quant(round_f16_t(
            round_f16_t(qkv3[:, :, :32] * q_norm) * scale_half.view(1, heads, 1)))
        k_pub = fp8_quant(round_f16_t(qkv3[:, :, 32:64] * k_norm))
        v_pub = fp8_quant(qkv3[:, :, 64:96])

        if ntr is not None:
            # One launch from here: scores, expWeight, the softmax tree, the value fold, the publication,
            # with the window's field addressing (no gather/scatter) inside the kernel.
            qkv_pub = torch.cat([q_pub, k_pub, v_pub], dim=2)          # [rows, heads, 96]
            out = ntr.window_attention(qkv_pub, prior, width, height, heads, shift_x, shift_y)
            return out.reshape(rows, heads * 32)

        # The reference path: the same arithmetic a step at a time.

        # Window slots: natural token index = y * 8 + x inside the window; window index = wx * windows_y + wy.
        win_x = torch.arange(windows_x, device=DEVICE, dtype=torch.long) * 8 - shift_x
        win_y = torch.arange(windows_y, device=DEVICE, dtype=torch.long) * 8 - shift_y
        ox = win_x.view(-1, 1).expand(windows_x, windows_y).reshape(-1)      # [windows]
        oy = win_y.view(1, -1).expand(windows_x, windows_y).reshape(-1)
        slot_index = torch.arange(64, device=DEVICE, dtype=torch.long)
        field_x = ox.view(-1, 1) + (slot_index % 8).view(1, -1)              # [windows, 64]
        field_y = oy.view(-1, 1) + (slot_index // 8).view(1, -1)
        valid = (field_x >= 0) & (field_x < width) & (field_y >= 0) & (field_y < height)
        field_index = field_y.clamp(0, height - 1) * width + field_x.clamp(0, width - 1)

        windows = windows_x * windows_y
        slots = 64
        gather_index = field_index.view(windows, slots)
        slot_valid = valid.view(windows, slots)

        def gather(pub):     # pub [rows, heads, 32] -> [windows * heads, slots, 32], zeros off field
            out = pub[gather_index.reshape(-1)].view(windows, slots, heads, 32)
            out = out.permute(0, 2, 1, 3).contiguous() * slot_valid.view(windows, 1, slots, 1)
            return out.reshape(windows * heads, slots, 32)

        q_win = gather(q_pub)
        k_win = gather(k_pub)
        v_win = gather(v_pub)
        prior_b = prior.unsqueeze(0).expand(windows, heads, slots, slots).reshape(windows * heads,
                                                                                slots, slots)

        # The score matrix multiply carries the learned prior as its accumulator.
        score = fp8_dot_chain_batched(q_win, k_win.transpose(1, 2).contiguous(), prior_b)
        scores = exp_weight_t(score)

        physical = torch.tensor([inverse_tiled_token(p) for p in range(64)], dtype=torch.long,
                                device=DEVICE)
        total = self._softmax_tree(scores[:, :, physical])
        reciprocal = round_f16_t(1.0 / total.float())
        weights = fp8_quant(round_f16_t(scores * reciprocal.unsqueeze(-1)))

        # The value fold walks the keys in 4x4-tiled order, 16 per group.
        out = fp8_dot_chain_batched(weights[:, :, physical], v_win[:, physical, :])
        attended = fp8_quant(out).reshape(windows, heads, slots, 32)

        result = torch.zeros((rows, heads * 32), dtype=torch.float16, device=DEVICE)
        flat_index = gather_index.reshape(-1)
        flat_valid = slot_valid.reshape(-1)
        values = attended.permute(0, 2, 1, 3).reshape(windows * slots, heads * 32)
        result[flat_index[flat_valid]] = values[flat_valid]
        return result

    # ------------------------------------------------------------------------------------------------
    # The global ViT (shaders/vit.wgsl).
    # ------------------------------------------------------------------------------------------------

    def vit_normalize(self, qkv, scales, tokens, padded, heads):
        """q/k/v normalized and published: the query carries an extra sqrt(32) and the learned scale."""
        qkv3 = qkv.view(tokens, heads, 96)
        q_norm = self._cosine_norm(qkv3[:, :, :32])
        k_norm = self._cosine_norm(qkv3[:, :, 32:64])
        learned = round_f16_t(self.np32(scales))
        head_scale = self.head_scale_const()

        q = qkv3[:, :, :32]
        nq = round_f16_t(round_f16_t(round_f16_t(q * q_norm) * head_scale)
                         * learned.view(1, heads, 1))
        nk = round_f16_t(qkv3[:, :, 32:64] * k_norm)
        nv = qkv3[:, :, 64:96]
        normalized = torch.zeros((padded, heads, 96), dtype=torch.float16, device=DEVICE)
        normalized[:tokens, :, :32] = fp8_quant(nq)
        normalized[:tokens, :, 32:64] = fp8_quant(nk)
        normalized[:tokens, :, 64:96] = fp8_quant(nv)
        return normalized

    def vit_attend(self, normalized, tokens, padded, heads):
        """Attention over every token at once. The weights are published unnormalized and the reciprocal
        reaches the result through the value sum; padding tokens add `padding * exp(0)` to the denominator,
        which is subtracted once at the end."""
        q = normalized[:tokens, :, :32].permute(1, 0, 2).contiguous()      # [heads, tokens, 32]
        k_all = normalized[:, :, 32:64].permute(1, 2, 0).contiguous()      # [heads, 32, padded]
        v_all = normalized[:, :, 64:96].permute(1, 0, 2).contiguous()      # [heads, padded, 32]

        score = fp8_dot_chain_batched(q, k_all, None)
        scores = vit_exp_weight_t(score)                                   # [heads, tokens, padded]

        total = torch.zeros(scores.shape[:2], dtype=torch.float16, device=DEVICE)
        for base in range(0, padded, 64):
            total = round_f16_t(total + self._softmax_tree(scores[:, :, base:base + 64]))
        padding = padded - tokens
        if padding > 0:
            zero_weight = vit_exp_weight_t(torch.zeros(1, device=DEVICE))[0]
            correction = round_f16_t(zero_weight.float() * float(padding))
            total = round_f16_t(total - correction)
        reciprocal = round_f16_t(1.0 / total.float())

        weights = fp8_quant(scores)                                        # unnormalized, E4M3
        value = fp8_dot_chain_batched(weights, v_all, None)                # [heads, tokens, 32]
        out = round_f16_t(value * reciprocal.unsqueeze(-1))
        return fp8_quant(out).permute(1, 0, 2).contiguous().reshape(tokens, heads * 32)

    # ------------------------------------------------------------------------------------------------
    # One block.
    # ------------------------------------------------------------------------------------------------

    def temporaries(self, rows, channels, layout):
        hidden = layout['expert_count'] * 128 if layout['expert_ffn'] else layout['hidden']
        return {
            'ffn': self.zeros(rows, hidden),
            'ffn_narrow': self.zeros(rows, channels) if layout['expert_ffn'] else None,
        }

    def block(self, block, channels, width, height, layout, tensor, state,
              out_e4=True, out_half=False, ffn_skip_override=None, phase=None):
        """FFN -> QKV -> window attention -> projection, with the two scaled skips."""
        model = self.model
        residual = ffn_skip_override if ffn_skip_override is not None else state
        ffn_aux = self.np16(model.aux_vector(tensor, layout['ffn_cos_skip'], channels))

        if layout['expert_ffn']:
            experts = layout['expert_count']
            w2_base = layout['expand'] + experts * channels * 128
            w3_base = w2_base + experts * 128 * 32
            # The fused chain wins when the row grid is dense enough (roughly 16*ch rows); below that
            # the batched gemms' wider grids beat it, so small levels keep the step-by-step path.
            if (ntr is not None and getattr(ntr, 'FUSED_EXPERT', False)
                    and layout['hidden'] == 128 and channels in (64, 128, 256)
                    and state.shape[0] >= 16 * channels):
                # The expert chain in one launch: expand (E broadcast experts) -> narrow (E branches)
                # -> contract (seeded) -> qkv, the intermediates kept in shared memory. The contract's
                # E4 is this block's projection residual (the raw half is not needed here).
                rows = state.shape[0]
                w1 = self.np16(model.fp8_matrix(tensor, layout['expand'], experts * channels, 128,
                                                batch_k=channels))
                w2 = self.np16(model.fp8_matrix(tensor, w2_base, experts * 128, 32, batch_k=128))
                w3 = self.np16(model.fp8_matrix(tensor, w3_base, channels, channels))
                w4 = self.np16(model.fp8_matrix(tensor, layout['qkv'], channels, channels * 3))
                ffn_quantized = self.zeros(rows, channels)
                qkv = self.zeros(rows, channels * 3)
                ntr.block_ffn_expert(state, w1, w2, w3, w4, residual, ffn_aux,
                                     None, ffn_quantized, qkv)
                attended = self.window_attention(
                    qkv, model.relative_bias(tensor, layout['relative'], layout['heads']),
                    model.head_scales(tensor, layout['scale'], layout['heads']),
                    width, height, layout['heads'], phase)
                return self.gemm(
                    attended, model.fp8_matrix(tensor, layout['projection'], channels, channels),
                    k=channels, n=channels, residual=ffn_quantized,
                    aux=self.np16(model.aux_vector(tensor, layout['attn_cos_skip'], channels)),
                    e4=out_e4, half=out_half)
            ffn, _ = self.gemm(
                state, model.fp8_matrix(tensor, layout['expand'], experts * channels, 128,
                                        batch_k=channels),
                k=channels, n=128, batches=experts, broadcast=True, silu=True)
            ffn_narrow, _ = self.gemm(
                ffn, model.fp8_matrix(tensor, w2_base, experts * 128, 32, batch_k=128),
                k=128, n=32, batches=experts)
            ffn_quantized, ffn_residual = self.gemm(
                ffn_narrow, model.fp8_matrix(tensor, w3_base, channels, channels),
                k=channels, n=channels, residual=residual, aux=ffn_aux, e4=True, half=True)
        elif (ntr is not None and getattr(ntr, 'FUSED_BLOCK', False)
              and channels == 32 and layout['hidden'] == 128):
            # The 32-channel chain in one launch: expand (SiLU, published) -> contract (seeded) -> qkv,
            # the intermediates kept in shared memory. The contract's E4 is only needed by qkv, so it
            # never touches global memory; the raw half feeds the projection's residual.
            rows = state.shape[0]
            w1 = self.np16(model.fp8_matrix(tensor, layout['expand'], 32, 128))
            w2 = self.np16(model.fp8_matrix(tensor, layout['contract_weights'], 128, 32))
            w3 = self.np16(model.fp8_matrix(tensor, layout['qkv'], 32, 96))
            ffn_residual = self.zeros(rows, 32)
            qkv = self.zeros(rows, 96)
            ntr.block_ffn(state, w1, w2, w3, residual, ffn_aux, ffn_residual, None, qkv)
            attended = self.window_attention(
                qkv, model.relative_bias(tensor, layout['relative'], layout['heads']),
                model.head_scales(tensor, layout['scale'], layout['heads']),
                width, height, layout['heads'], phase)
            return self.gemm(
                attended, model.fp8_matrix(tensor, layout['projection'], channels, channels),
                k=channels, n=channels, residual=ffn_residual,
                aux=self.np16(model.aux_vector(tensor, layout['attn_cos_skip'], channels)),
                e4=out_e4, half=out_half)
        else:
            ffn, _ = self.gemm(
                state, model.fp8_matrix(tensor, layout['expand'], channels, layout['hidden']),
                k=channels, n=layout['hidden'], silu=True)
            ffn_quantized, ffn_residual = self.gemm(
                ffn, model.fp8_matrix(tensor, layout['contract_weights'], layout['hidden'], channels),
                k=layout['hidden'], n=channels, residual=residual, aux=ffn_aux, e4=True, half=True)

        _, qkv = self.gemm(
            ffn_quantized, model.fp8_matrix(tensor, layout['qkv'], channels, channels * 3),
            k=channels, n=channels * 3, e4=False, half=True)
        attended = self.window_attention(
            qkv, model.relative_bias(tensor, layout['relative'], layout['heads']),
            model.head_scales(tensor, layout['scale'], layout['heads']),
            width, height, layout['heads'], phase)

        return self.gemm(
            attended, model.fp8_matrix(tensor, layout['projection'], channels, channels),
            k=channels, n=channels,
            residual=ffn_quantized if layout['expert_ffn'] else ffn_residual,
            aux=self.np16(model.aux_vector(tensor, layout['attn_cos_skip'], channels)),
            e4=out_e4, half=out_half)

    def split_block(self, block, width, height, state, out_e4=True, out_half=False, phase=None):
        """The 512 stage: eight independent 64-wide branches widened to 256 and back, over one 512 -> 512."""
        model = self.model
        channels, branches, branch_channels, middle_channels, heads = 512, 8, 64, 256, 16
        branch_tensor = model.tensor(block, 0)
        contract = model.tensor(block, 1)
        qkv_tensor = model.tensor(block, 2)
        projection = model.tensor(block, 3)
        w2_base = branches * channels * branch_channels
        w3_base = w2_base + branches * branch_channels * middle_channels
        qkv_relative = channels * channels * 3
        qkv_scale = qkv_relative + heads * 8192

        branch, _ = self.gemm(state, model.fp8_matrix(branch_tensor, 0, channels, channels),
                              k=channels, n=channels)
        middle, _ = self.gemm(
            branch, model.fp8_matrix(branch_tensor, w2_base, branches * branch_channels,
                                     middle_channels, batch_k=branch_channels),
            k=branch_channels, n=middle_channels, batches=branches, silu=True)
        layer0, _ = self.gemm(
            middle, model.fp8_matrix(branch_tensor, w3_base, branches * middle_channels,
                                     branch_channels, batch_k=middle_channels),
            k=middle_channels, n=branch_channels, batches=branches)
        ffn_residual, _ = self.gemm(
            layer0, model.fp8_matrix(contract, 0, channels, channels),
            k=channels, n=channels, residual=state,
            aux=self.np16(model.aux_vector(contract, channels * channels, channels)))
        _, qkv = self.gemm(ffn_residual, model.fp8_matrix(qkv_tensor, 0, channels, channels * 3),
                           k=channels, n=channels * 3, e4=False, half=True)
        attended = self.window_attention(
            qkv, model.relative_bias(qkv_tensor, qkv_relative, heads),
            model.head_scales(qkv_tensor, qkv_scale, heads), width, height, heads, phase)
        return self.gemm(
            attended, model.fp8_matrix(projection, 0, channels, channels),
            k=channels, n=channels, residual=ffn_residual,
            aux=self.np16(model.aux_vector(projection, channels * channels, channels)),
            e4=out_e4, half=out_half)

    def vit(self, state, tokens):
        """Eight blocks whose attention spans every token of the coarsest level."""
        channels, heads, ffn_channels = 1024, 32, 4096
        padded = self.g['padded_vit_tokens']
        model = self.model
        for block in range(31, 39):
            expand = model.tensor(block, 0)
            contract = model.tensor(block, 1)
            qkv_tensor = model.tensor(block, 2)
            projection = model.tensor(block, 4)
            expanded, _ = self.gemm(
                state, model.fp8_matrix(expand, 0, channels, ffn_channels),
                k=channels, n=ffn_channels, silu=True)
            ffn_residual, _ = self.gemm(
                expanded, model.fp8_matrix(contract, 0, ffn_channels, channels),
                k=ffn_channels, n=channels, partition=1024, residual=state,
                aux=self.np16(model.aux_vector(contract, ffn_channels * channels, channels)))
            # The ViT's qkv tensor puts its per-head scales before the weights, where every other block
            # puts them after. Nothing depends on that but the offsets.
            _, qkv = self.gemm(
                ffn_residual, model.fp8_matrix(qkv_tensor, heads * 4, channels, channels * 3),
                k=channels, n=channels * 3, partition=512, e4=False, half=True)
            if ntr is not None and getattr(ntr, 'FUSED_VIT', False):
                # One launch from the raw qkv: the cosine norm, the q/k/v publications (q carries the
                # sqrt(32) and the learned scale), the scores, the chunked softmax trees with the
                # padding correction, and the value chain are all inside the kernel.
                learned = round_f16_t(self.np32(model.head_scales(qkv_tensor, 0, heads)))
                attended = ntr.vit_attention_raw(
                    qkv.view(tokens, heads, 96), learned, self.head_scale_const(),
                    tokens, padded, heads).reshape(tokens, heads * 32)
            else:
                normalized = self.vit_normalize(qkv, model.head_scales(qkv_tensor, 0, heads),
                                                tokens, padded, heads)
                attended = self.vit_attend(normalized, tokens, padded, heads)
            state, _ = self.gemm(
                attended, model.fp8_matrix(projection, 0, channels, channels),
                k=channels, n=channels, partition=256, residual=ffn_residual,
                aux=self.np16(model.aux_vector(projection, channels * channels, channels)))
            self.capture(f'block-{block}', state)
        return state

    # ------------------------------------------------------------------------------------------------
    # The whole network. `features` is f32 [fullRows][16]; the head is f32 [fullRows][4].
    # ------------------------------------------------------------------------------------------------

    def record(self, features):
        self.phases = WindowPhases()
        self.boundaries = {}
        g = self.g
        model = self.model
        full_rows = g['full_rows']
        d0, d1, d2, d3, d4, d5 = g['levels']

        # ---- Block 0 at full resolution, behind the f16 input adapter.
        pre_tensor = model.tensor(0)
        pre_layout = pre_fused_layout()
        if pre_tensor.byte_length != pre_layout['end_without_padding'] + 16:
            raise ValueError('unexpected block 0 layout')
        adapter_w = model.f16_matrix(pre_tensor, pre_layout['input_adapter'], 16, 32)
        adapter_e4, adapter_f16, _ = self.gemm_f16(features.to(torch.float16), adapter_w, 16, 32,
                                                   e4=True, half=True)
        block0, block0_raw = self.block(
            0, 32, g['full_width'], g['full_height'], pre_layout, pre_tensor, adapter_e4,
            out_e4=True, out_half=True, ffn_skip_override=adapter_f16, phase=self.phases.take(6))
        self.capture('block-0', block0)

        # ---- Down to level 0 and the four 32-channel blocks there.
        state = self.downsample(block0_raw, 32, g['full_width'], g['full_height'],
                                d0['width'], d0['height'])
        self.capture('transition-0-1', state)
        level0_raw = None
        layout0 = fused_layout(32)
        for block in range(1, 5):
            state, raw = self.block(
                block, 32, d0['width'], d0['height'], layout0, model.tensor(block), state,
                out_e4=True, out_half=block == 4, phase=self.phases.take(0))
            if block == 4:
                level0_raw = raw
            self.capture(f'block-{block}', state)
        skip32 = state

        # ---- Encoder stages 64 / 128 / 256, each ending in a pool and a widening transition.
        pooled32 = self.downsample(level0_raw, 32, d0['width'], d0['height'], d1['width'], d1['height'])
        self.capture('pooled-4-5', pooled32)
        stage_input, _ = self.gemm(
            pooled32, model.fp8_matrix(model.tensor(4), layout0['end_without_padding'], 32, 64),
            k=32, n=64)
        self.capture('transition-4-5', stage_input)

        encoder_stages = [
            {'level': d1, 'next': d2, 'channels': 64, 'first': 5, 'last': 8, 'level_index': 1},
            {'level': d2, 'next': d3, 'channels': 128, 'first': 9, 'last': 14, 'level_index': 2},
            {'level': d3, 'next': d4, 'channels': 256, 'first': 15, 'last': 22, 'level_index': 3},
        ]
        skips = []
        for stage in encoder_stages:
            layout = fused_layout(stage['channels'])
            st = stage_input
            raw = None
            for block in range(stage['first'], stage['last'] + 1):
                st, r = self.block(
                    block, stage['channels'], stage['level']['width'], stage['level']['height'],
                    layout, model.tensor(block), st, out_e4=True, out_half=block == stage['last'],
                    phase=self.phases.take(stage['level_index']))
                if block == stage['last']:
                    raw = r
                self.capture(f'block-{block}', st)
            skips.append(st)
            pooled = self.downsample(raw, stage['channels'], stage['level']['width'],
                                     stage['level']['height'], stage['next']['width'],
                                     stage['next']['height'])
            self.capture(f"pooled-{stage['last']}-{stage['last'] + 1}", pooled)
            stage_input, _ = self.gemm(
                pooled, model.fp8_matrix(model.tensor(stage['last']), layout['end_without_padding'],
                                         stage['channels'], stage['channels'] * 2),
                k=stage['channels'], n=stage['channels'] * 2)
            self.capture(f"transition-{stage['last']}-{stage['last'] + 1}", stage_input)
        skip64, skip128, skip256 = skips

        # ---- The 512 stage, the pool into the ViT, and the ViT.
        st = stage_input
        raw = None
        for block in range(23, 31):
            st, r = self.split_block(block, d4['width'], d4['height'], st, out_e4=True,
                                     out_half=block == 30, phase=self.phases.take(4))
            if block == 30:
                raw = r
            self.capture(f'block-{block}', st)
        skip512 = st
        tokens = g['vit_tokens']
        pooled = self.downsample(raw, 512, d4['width'], d4['height'], d5['width'], d5['height'])
        vit_state, _ = self.gemm(pooled, model.fp8_matrix(model.tensor(30, 4), 0, 512, 1024),
                                 k=512, n=1024)
        vit_state = self.vit(vit_state, tokens)

        # ---- Decoder 512: the ViT output projected, doubled, and merged onto the encoder skip.
        _, projected = self.gemm(vit_state, model.fp8_matrix(model.tensor(39), 0, 1024, 512),
                                 k=1024, n=512, partition=256, e4=False, half=True)
        dst = self.upsample_residual(
            projected, skip512, self.np16(model.aux_vector(model.tensor(39), 1024 * 512, 512)),
            512, d5['width'], d5['height'], d4['width'], d4['height'])[0]
        self.capture('block-39', dst)
        for block in range(40, 48):
            dst, _ = self.split_block(block, d4['width'], d4['height'], dst, out_e4=True,
                                      phase=self.phases.take(4))
            self.capture(f'block-{block}', dst)

        # ---- Decoder stages 256 / 128 / 64 / 32.
        decoder_stages = [
            {'low': d4, 'high': d3, 'channels': 256, 'first': 48, 'last': 55,
             'level_index': 3, 'skip': skip256},
            {'low': d3, 'high': d2, 'channels': 128, 'first': 56, 'last': 61,
             'level_index': 2, 'skip': skip128},
            {'low': d2, 'high': d1, 'channels': 64, 'first': 62, 'last': 65,
             'level_index': 1, 'skip': skip64},
            {'low': d1, 'high': d0, 'channels': 32, 'first': 66, 'last': 69,
             'level_index': 0, 'skip': skip32},
        ]
        for stage in decoder_stages:
            channels = stage['channels']
            transition = model.tensor(stage['first'])
            layout = upsample_fused_layout(channels * 2, channels)
            if transition.byte_length != layout['end_without_padding'] + 16:
                raise ValueError(f"unexpected upsample layout for block {stage['first']}")
            _, projection = self.gemm(
                dst, model.fp8_matrix(transition, layout['upsample_weight'], channels * 2, channels),
                k=channels * 2, n=channels, e4=False, half=True)
            st, merged_raw = self.upsample_residual(
                projection, stage['skip'],
                self.np16(model.aux_vector(transition, layout['transition_scale'], channels)),
                channels, stage['low']['width'], stage['low']['height'],
                stage['high']['width'], stage['high']['height'], dual=channels == 32)
            for block in range(stage['first'], stage['last'] + 1):
                st, _ = self.block(
                    block, channels, stage['high']['width'], stage['high']['height'],
                    layout if block == stage['first'] else fused_layout(channels),
                    model.tensor(block), st, out_e4=True,
                    ffn_skip_override=merged_raw if block == stage['first'] else None,
                    phase=self.phases.take(stage['level_index']))
                self.capture(f'block-{block}', st)
            dst = st

        # ---- The post block at full resolution, and the head.
        tensor = model.tensor(70)
        layout = post_fused_layout()
        if tensor.byte_length != layout['end_without_padding']:
            raise ValueError('unexpected block 70 layout')
        aux_pair = self.np16(model.aux_pair(tensor, layout['input_scale'], layout['adapter_scale'], 32))
        merged, merged_raw = self.post_blend(
            dst, block0, aux_pair, 32, d0['width'], d0['height'], g['full_width'], g['full_height'])
        _, block_raw = self.block(
            70, 32, g['full_width'], g['full_height'], layout, tensor, merged,
            out_e4=False, out_half=True, ffn_skip_override=merged_raw, phase=self.phases.take(6))
        head_w = model.f16_matrix(tensor, layout['post_weights'], 32, 4)
        _, _, head = self.gemm_f16(block_raw, head_w, 32, 4, e4=False, half=False, f32=True)
        return head
