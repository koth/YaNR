"""Versioned contract and exact parameter ledger for the shared/routed prototype.

Pure stdlib: candidates can be audited before allocating GPU/torch weights.
2026-10-18:去掉共享专家 —— bank 轴 0..15 全是路由专家,每 token 激活 top-2。
原设计里轴 0 是可训练共享专家、轴 1..16 是路由专家,融合为
(shared + g·routed)·0.5;实测该共享支路在 d3/d4 占到 FFN 输出的 62-74%
却不专门化,因此移除,两路都改为被路由的专家。
"""
import copy
import math

ARCHITECTURE = 'shared-routed-moe-v1'
KIND = 'shared_routed_moe'
LEVELS = ('full', 'd0', 'd1', 'd2', 'd3', 'd4')
ROUTING = dict(shared_experts=0, routed_experts=7, top_k=1,
               spatial_granularity='window8', vit_granularity='token',
               selection='fp32-argmax-with-bias-v1', tie_break='lowest-index',
               gate='softmax-selected-times-routed-v1',
               fusion='sub-layer-mean-single-attention-v1', fusion_scale=0.5,
               jitter=0.0, transitions='expertized')


def is_shared(shape):
    return shape.get('architecture') == ARCHITECTURE


def validate(shape):
    if shape.get('architecture') != ARCHITECTURE:
        raise ValueError('unsupported shared/routed architecture version')
    if shape.get('routing') != ROUTING:
        raise ValueError('shared/routed routing contract is missing or incompatible')
    levels = shape.get('levels', [])
    if [v.get('level') for v in levels] != list(LEVELS):
        raise ValueError('shared/routed levels must be full,d0,d1,d2,d3,d4 in order')
    for level in levels:
        if level.get('kind') != KIND or level.get('window_slots', 8) != 8:
            raise ValueError('shared/routed requires its own kind and 8x8 windows')
        for key in ('channels', 'hidden'):
            value = level.get(key)
            if type(value) is not int or value < 32 or value > 1024 or value % 32:
                raise ValueError(f"{level['level']}.{key}: expected 32..1024, multiple of 32")
        if type(level.get('blocks')) is not int or level['blocks'] < 2:
            raise ValueError('each spatial level needs at least two blocks')
        if type(level.get('attn')) is not bool:
            raise ValueError('each spatial level needs explicit attn boolean')
        if 'experts' in level and level['experts'] != 7:
            raise ValueError('shared/routed banks always contain exactly 16 routed experts')
    vit = shape.get('vit', {})
    if vit.get('kind') != KIND or vit.get('tokens_level') != 'd5':
        raise ValueError('shared/routed ViT kind/tokens_level mismatch')
    for key in ('channels', 'ffn'):
        value = vit.get(key)
        if type(value) is not int or value < 32 or value % 32:
            raise ValueError('ViT channels/ffn must be positive multiples of 32')
    # 2026-10-18:允许 blocks=0 —— 即完全去掉 ViT 块,只保留 vit_in/vit_out 的降采样通路。
    # 动机:实测 v4.5 的 3 个 ViT 块占了 83% 参数(66.45M),但 forward hook 消融显示
    # 去掉它们后 8-64px 的相关变化在 ±0.01 内(只丢最低频 0.11)。要验证"这些参数是不是
    # 白放在这里",最直接的做法就是把块数设 0 重训一次,而不是继续在空间部分微调。
    if type(vit.get('blocks')) is not int or vit['blocks'] < 0:
        raise ValueError('ViT block count must be a non-negative integer')
    if vit['channels'] > 1024:
        raise ValueError('shared/routed router supports at most 1024 input channels')
    if 'experts' in vit and vit['experts'] != 7:
        raise ValueError('shared/routed ViT always contains exactly 16 routed experts')
    return shape


def candidate(channels, hidden, vit_channels, vit_ffn, blocks=(2, 4, 4, 2, 2, 2), vit_blocks=3):
    return validate(dict(name='student_shared_moe_proto', architecture=ARCHITECTURE,
                         routing=copy.deepcopy(ROUTING),
                         levels=[dict(level=n, kind=KIND, channels=c, hidden=h, blocks=b,
                                      attn=n != 'full')
                                 for n, c, h, b in zip(LEVELS, channels, hidden, blocks)],
                         vit=dict(kind=KIND, channels=vit_channels, ffn=vit_ffn,
                                  blocks=vit_blocks, tokens_level='d5')))


def ledger(shape):
    """Named state shapes exactly matching StudentNetwork (including FP32 buffers)."""
    validate(shape)
    entries = []

    def add(name, dims, parameter=True):
        entries.append(dict(name=name, shape=list(dims), dtype='f32',
                            byteLength=4 * math.prod(dims), isParameter=parameter))

    def router(prefix, channels):
        add(prefix + '.router.0.weight', [channels])
        add(prefix + '.router.0.bias', [channels])
        add(prefix + '.router.1.weight', [7, channels])
        add(prefix + '.router_bias', [7], False)

    def transition(prefix, ci, co):
        router(prefix, ci)
        add(prefix + '.weight', [7, ci, co])
        add(prefix + '.bias', [7, co])

    def block(prefix, c, h, attn, vit=False):
        router(prefix, c)
        add(prefix + '.expand', [7, c, h])
        add(prefix + ('.contract' if vit else '.narrow'), [7, h, c])
        add(prefix + '.aux_ffn', [c])
        if attn:
            add(prefix + '.qkv', [7, c, 3 * c])
            add(prefix + '.proj', [7, c, c])
            add(prefix + '.aux_attn', [c])
            add(prefix + '.attention.scale', [c // 32])
            if not vit:
                add(prefix + '.attention.prior', [c // 32, 64, 64])

    channels = {v['level']: v['channels'] for v in shape['levels']}
    cf = channels['full']
    add('blend_scale', [1])
    add('adapter.weight', [cf, 16]); add('adapter.bias', [cf])
    add('head.weight', [4, cf]); add('head.bias', [4])
    for level in shape['levels']:
        enc = max(1, level['blocks'] // 2)
        for branch, count in [('enc_blocks', enc), ('dec_blocks', level['blocks'] - enc)]:
            for j in range(count):
                block(f"{branch}.{level['level']}.{j}", level['channels'], level['hidden'], level['attn'])
    for low, high in zip(LEVELS, LEVELS[1:]):
        transition('enc_trans.' + high, channels[low], channels[high])
    for low, high in [('d3', 'd4'), ('d2', 'd3'), ('d1', 'd2'), ('d0', 'd1')]:
        transition('dec_trans.' + low, channels[high], channels[low])
    for name in LEVELS[1:]:
        add('merge.' + name + '.aux', [channels[name]])
    v = shape['vit']
    transition('vit_in', channels['d4'], v['channels'])
    transition('vit_out', v['channels'], channels['d4'])
    for j in range(v['blocks']):
        block(f'vit_blocks.{j}', v['channels'], v['ffn'], True, True)
    add('post.aux_pair', [2, cf])
    return entries
