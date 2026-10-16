"""Detailed arithmetic accounting, separate from the calibrated legacy estimator.

Attention is the padded mathematical reference, not measured kernel instruction
counts. Group alignment overhead requires measured per-expert row loads; absent
loads it is unknown, never silently zero or an equal-capacity approximation.
"""
from nr_geometry import window_phase
from moe_shape import LEVELS, is_shared, validate


def account_moe(shape, g, loads=None, tile_rows=32):
    if is_shared(shape):
        validate(shape)
    elif any(lv['kind'] != 'moe' for lv in shape['levels']):
        raise ValueError('detailed sparse accounting requires all-MoE or shared/routed shape')
    shared = is_shared(shape)
    passes = 2 if shared else 1
    layers = []
    nonmac = dict(router_input_elements=0, router_logits=0, packed_row_elements=0,
                  silu_elements=0, attention_score_elements=0, gate_fusion_elements=0)

    def size(name):
        if name == 'full':
            return g['full_width'], g['full_height']
        lv = g['levels'][int(name[1:])]
        return lv['width'], lv['height']

    def add(name, width, height, ci, co=None, hidden=None, attn=False, phase=0, token=False, routed=True):
        rows = width * height
        units = rows if token else ((width + 7) // 8) * ((height + 7) // 8)
        experts = 0
        if routed:
            experts = 16 if shared else int(shape['vit'].get('experts', 8) if token else
                                           next(v['experts'] for v in shape['levels'] if v['level'] == name.split('.')[1]))
        matrices = [(ci, co)] if co is not None else [(ci, hidden), (hidden, ci)] + ([(ci, 3 * ci), (ci, ci)] if attn else [])
        projection = sum(k * n for k, n in matrices)
        padded_projection = sum(k * (n if n == 32 else (n + 63) // 64 * 64) for k, n in matrices)
        route_macs = units * ci * experts if routed else 0
        attention = 0
        scores = 0
        if attn:
            if token:
                padded = (rows + 63) & ~63
                scores = (ci // 32) * rows * padded
            else:
                sx, sy = window_phase(phase)
                windows = ((width + sx + 7) // 8) * ((height + sy + 7) // 8)
                scores = (ci // 32) * windows * 64 * 64
            attention = scores * 64  # QK and AV, head_dim=32
        actual_load = None if loads is None else loads.get(name)
        alignment = None
        if actual_load is not None:
            if len(actual_load) != experts or any(type(v) is not int or v < 0 for v in actual_load) or sum(actual_load) != rows:
                raise ValueError('measured row load mismatch: ' + name)
            alignment = sum((-n) % tile_rows for n in actual_load if n)
        dense_passes = passes if routed else 1
        row = dict(name=name, rows=rows, router_units=units if routed else 0,
                   shared_linear_macs=rows * projection if shared and routed else 0,
                   routed_linear_macs=rows * projection if routed else 0,
                   public_linear_macs=0 if routed else rows * projection,
                   router_macs=route_macs, attention_padded_reference_macs=attention,
                   routed_load_rows=actual_load, alignment_extra_rows=alignment,
                   row_alignment_extra_macs=None if alignment is None else alignment * projection,
                   column_alignment_extra_macs=None if alignment is None else (rows + alignment) * (padded_projection - projection),
                   alignment_extra_macs=None if alignment is None else (rows + alignment) * padded_projection - rows * projection,
                   routed_matrices=[dict(inputs=k, outputs=n, tile_columns=32 if n == 32 else 64) for k, n in matrices] if routed else [],
                   linear_macs=rows * projection * dense_passes)
        row['total_macs'] = row['linear_macs'] + attention + route_macs
        layers.append(row)
        if routed:
            nonmac['router_input_elements'] += units * ci
            nonmac['router_logits'] += units * experts
            # One gather per sublayer input; inverse scatter/fusion also touches these rows.
            nonmac['packed_row_elements'] += rows * (ci if co is not None else ci + hidden + (4 * ci if attn else 0))
            nonmac['gate_fusion_elements'] += rows * (co if co is not None else ci + (4 * ci if attn else 0))
        if hidden is not None:
            nonmac['silu_elements'] += rows * hidden * dense_passes
        nonmac['attention_score_elements'] += scores

    ch = {v['level']: v['channels'] for v in shape['levels']}
    for lv in shape['levels']:
        name = lv['level']; enc = max(1, lv['blocks'] // 2)
        for j in range(lv['blocks']):
            branch, index = ('enc_blocks', j) if j < enc else ('dec_blocks', j - enc)
            add(f'{branch}.{name}.{index}', *size(name), lv['channels'], hidden=lv['hidden'], attn=lv['attn'], phase=j)
    for low, high in zip(LEVELS, LEVELS[1:]):
        add('enc_trans.' + high, *size(high), ch[low], co=ch[high], routed=shared)
    for low, high in [('d3', 'd4'), ('d2', 'd3'), ('d1', 'd2'), ('d0', 'd1')]:
        # Decoder projects BEFORE upsampling at the higher level's row count.
        add('dec_trans.' + low, *size(high), ch[high], co=ch[low], routed=shared)
    vit = shape['vit']
    add('vit_in', *size('d5'), ch['d4'], co=vit['channels'], token=True, routed=shared)
    add('vit_out', *size('d5'), vit['channels'], co=ch['d4'], token=True, routed=shared)
    for j in range(vit['blocks']):
        add(f'vit_blocks.{j}', *size('d5'), vit['channels'], hidden=vit['ffn'], attn=True, token=True)
    add('adapter', *size('full'), 16, co=ch['full'], routed=False)
    add('head', *size('full'), ch['full'], co=4, routed=False)
    return dict(total_macs=sum(l['total_macs'] for l in layers), layers=layers,
                nonmac_work=nonmac, tile_rows=tile_rows,
                note='Ideal actual-row expert MACs; padding costs require measured loads. Attention uses padded reference geometry. No latency prediction.')
