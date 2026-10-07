# The shapes the graph is built on: the padded field, the six pooling levels, the window phase cycle, and the
# byte layout of a block's weight tensor. A line-for-line port of ports/browser-webgpu/src/geometry.js, which
# mirrors src/nr_graph.cpp. None of this is a free choice - the field size decides which tokens exist, and
# therefore the result inside the valid rectangle as well.


def align_up(value, alignment):
    return -(-value // alignment) * alignment


def field_alignment(valid):
    """
    Every level halves its input and rounds up to 4, and the decoder doubles the chain back up, so a dimension
    needs enough headroom for all of those halvings to be exact. Six are the halvings; level 0 adds a seventh
    when it is not a whole number of 8-pixel windows.
    """
    reductions = 0
    size = valid
    for level in range(6):
        half = align_up((size + 1) // 2, 4)
        if half < size:
            reductions += 1
        if level == 0 and half % 8 != 0:
            reductions += 1
        size = half
    return 1 << reductions


def geometry_from_valid(valid_width, valid_height):
    align_width = field_alignment(valid_width)
    align_height = field_alignment(valid_height)
    full_width = max(320, align_up(valid_width, align_width))
    full_height = max(320, align_up(valid_height, align_height))
    # One more alignment step on the width when both axes are a multiple of four alignments. The rule has no
    # stated reason; it has to be reproduced because it moves the window grid.
    if full_width % (4 * align_width) == 0 and full_height % (4 * align_height) == 0:
        full_width += align_width

    levels = []
    width, height = full_width, full_height
    for _ in range(6):
        width = align_up((width + 1) // 2, 4)
        height = align_up((height + 1) // 2, 4)
        levels.append({'width': width, 'height': height, 'rows': width * height})
    if levels[0]['width'] % 8 or levels[0]['height'] % 8:
        raise ValueError(f'unsupported size {valid_width}x{valid_height}: level 0 '
                         f"({levels[0]['width']}x{levels[0]['height']}) is not a whole number of 8-pixel "
                         'windows; use at least 33 pixels on each axis')
    vit_tokens = levels[5]['rows']
    return {
        'valid_width': valid_width, 'valid_height': valid_height,
        'full_width': full_width, 'full_height': full_height,
        'levels': levels,
        'full_rows': full_width * full_height,
        'vit_tokens': vit_tokens,
        'padded_vit_tokens': (vit_tokens + 63) & ~63,
    }


# The four window views, as origin offsets (-shiftX, -shiftY). Every resolution level runs its own cycle, one
# step per block at that level in visit order, and a decoder stage continues the count its encoder stage left.
PHASES = [[0, 0], [4, 4], [4, 0], [0, 4]]


def window_phase(index):
    return PHASES[index & 3]


class WindowPhases:
    def __init__(self):
        self.counters = [0] * 7

    def take(self, level):
        value = self.counters[level]
        self.counters[level] += 1
        return value

    def reset(self):
        self.counters = [0] * 7


def standard_hidden(channels):
    if channels in (32, 64, 128, 256):
        return 128
    raise ValueError(f'no fused layout for {channels} channels')


def fused_layout(channels, base=0):
    """Byte offsets inside one block's weight tensor: FFN -> QKV -> window attention -> projection."""
    hidden = standard_hidden(channels)
    heads = channels // 32
    expert_ffn = channels >= 64
    expert_count = channels // 32 if expert_ffn else 0
    expand_bytes = expert_count * channels * 128 if expert_ffn else channels * hidden
    ffn_weight_bytes = (expand_bytes + expert_count * 128 * 32 + expert_count * 32 * channels) if expert_ffn \
        else expand_bytes + hidden * channels
    layout = {'hidden': hidden, 'heads': heads, 'expert_ffn': expert_ffn, 'expert_count': expert_count,
              'expand': base, 'contract_weights': base + expand_bytes}
    layout['ffn_cos_skip'] = base + ffn_weight_bytes + 16
    layout['qkv'] = layout['ffn_cos_skip'] + channels * 2 + 16
    layout['relative'] = layout['qkv'] + channels * channels * 3
    layout['scale'] = layout['relative'] + heads * 8192
    layout['projection'] = layout['scale'] + align_up(heads * 4, 16)
    layout['attn_cos_skip'] = layout['projection'] + channels * channels
    layout['end_without_padding'] = layout['attn_cos_skip'] + channels * 2
    return layout


def pre_fused_layout():
    """Block 0: the same block with a 16 -> 32 f16 input adapter in front of it."""
    return {
        'hidden': 128, 'heads': 1, 'expert_ffn': False, 'expert_count': 0,
        'expand': 0, 'contract_weights': 4096, 'input_adapter': 8208, 'ffn_cos_skip': 9232, 'qkv': 9312,
        'relative': 12384, 'scale': 20576, 'projection': 20592, 'attn_cos_skip': 21616,
        'end_without_padding': 21680,
    }


def upsample_fused_layout(input_channels, channels):
    """The first block of a decoder stage: the 2x upsample weight and the skip scale sit before the QKV."""
    if input_channels != channels * 2:
        raise ValueError('upsample layout expects 2x input channels')
    hidden = standard_hidden(channels)
    heads = channels // 32
    narrow_padding = 16 if channels == 32 else 0
    expert_ffn = channels >= 64
    expert_count = channels // 32 if expert_ffn else 0
    expand_bytes = expert_count * channels * 128 if expert_ffn else channels * hidden
    ffn_weight_bytes = (expand_bytes + expert_count * 128 * 32 + expert_count * 32 * channels) if expert_ffn \
        else expand_bytes + hidden * channels
    layout = {'hidden': hidden, 'heads': heads, 'expert_ffn': expert_ffn, 'expert_count': expert_count,
              'expand': 0, 'contract_weights': expand_bytes}
    layout['upsample_weight'] = ffn_weight_bytes
    layout['ffn_cos_skip'] = layout['upsample_weight'] + input_channels * channels + narrow_padding
    layout['transition_scale'] = layout['ffn_cos_skip'] + channels * 2 + narrow_padding
    layout['qkv'] = layout['transition_scale'] + channels * 2
    layout['relative'] = layout['qkv'] + channels * channels * 3
    layout['scale'] = layout['relative'] + heads * 8192
    layout['projection'] = layout['scale'] + align_up(heads * 4, 16)
    layout['attn_cos_skip'] = layout['projection'] + channels * channels
    layout['end_without_padding'] = layout['attn_cos_skip'] + channels * 2
    return layout


def post_fused_layout():
    """Block 70: the post blend's two scales in front, the 32 -> 4 f16 head at the end."""
    return {
        'hidden': 128, 'heads': 1, 'expert_ffn': False, 'expert_count': 0,
        'expand': 0, 'contract_weights': 4096, 'ffn_cos_skip': 8208, 'input_scale': 8272,
        'adapter_scale': 8336, 'qkv': 8400, 'relative': 11472, 'scale': 19664, 'projection': 19680,
        'attn_cos_skip': 20704, 'post_weights': 20784, 'end_without_padding': 21808,
    }
