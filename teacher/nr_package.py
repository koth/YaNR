#!/usr/bin/env python3
"""学生模型打包/加载(7.1/7.2/7.3):与教师同格式的 manifest.json + model/stages/s*.bin。

教师格式(nr_model.py / docs/weights.md):manifest.json 的 stages 段(逐文件 sha256)+
tensors 段(逐张量 stage/stageOffset/byteLength)。学生包同一骨架,差异只有三点:

  1. 张量是 **f32 小端裸数据**(教师是 DLL 打包记录,无需解包);
  2. tensors 条目多带 shape/dtype(教师靠 block 语义定形,学生按名字重建 state_dict);
  3. model 段记录 kind='student' + 形状名/训练步数/blend_scale/版本串,shape.json(形状 DSL
     原样内嵌)随包走 —— 加载端自动探测,零外部文件依赖。

    写:convert_weights.py --shape student ckpt.pt -o weights/student_mixed
    读:load_student_package(dir, width, height, device) -> (StudentNetwork, manifest)

打包校验(7.3)见 check_package.py:包内重建的模型与训练 ckpt 的 torch 前向逐位一致。
"""
import hashlib
import json
import os
import re
from collections import OrderedDict

import numpy as np
import torch

from nr_geometry import geometry_from_valid
from nr_student import StudentNetwork, load_student_shape

BLOCK_NAME = re.compile(r'^((?:enc_blocks|dec_blocks)\.[A-Za-z0-9_]+\.\d+|vit_blocks\.\d+)\.(.+)$')


def _block_order(model):
    """块的稳定顺序(与 dump_weights 的 blocks 遍历一致),用于 manifest 的 block 编号。"""
    order = []
    for side in ('enc_blocks', 'dec_blocks'):
        for level, module_list in getattr(model, side).items():
            for i in range(len(module_list)):
                order.append(f'{side}.{level}.{i}')
    for i in range(len(model.vit_blocks)):
        order.append(f'vit_blocks.{i}')
    return {key: idx for idx, key in enumerate(order)}


def model_version(shape_name, step):
    return f'student-{shape_name}-step{step}' if step >= 0 else f'student-{shape_name}'


def write_package(model, state, out_dir, shape_path, step=-1, stages=3, extra=None):
    """state(state_dict,f32)→ out_dir/{manifest.json, model/stages/s*.bin, shape.json}。

    tensors 条目与教师同键(name/block/layer/parameter/stage/stageOffset/byteLength),
    另加 shape/dtype;块外张量(头部/适配层)block=-1。
    """
    block_idx = _block_order(model)
    shape_name = os.path.splitext(os.path.basename(shape_path))[0]

    records = []
    for name, tensor in state.items():
        arr = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
        match = BLOCK_NAME.match(name)
        block, parameter = (-1, name) if not match else (block_idx[match.group(1)], match.group(2))
        records.append((name, block, parameter, arr))

    total = sum(arr.nbytes for *_, arr in records)
    per_stage = (total + stages - 1) // stages
    stage_list, tensors = [], []
    current = None
    for name, block, parameter, arr in records:
        raw = arr.tobytes()                                  # 小端 f32(平台即 x86/ARM-LE)
        if current is None or (len(current[1]) >= per_stage and len(stage_list) < stages):
            current = (str(len(stage_list)), bytearray())
            stage_list.append(current)
        stage_id, buffer = current
        tensors.append({
            'name': name, 'block': block, 'layer': 0, 'parameter': parameter,
            'stage': stage_id, 'stageOffset': len(buffer), 'byteLength': len(raw),
            'shape': list(arr.shape), 'dtype': 'f32',
        })
        buffer += raw

    os.makedirs(os.path.join(out_dir, 'model', 'stages'), exist_ok=True)
    manifest_stages = []
    for stage_id, buffer in stage_list:
        rel = f'stages/s{stage_id}.bin'
        with open(os.path.join(out_dir, 'model', rel), 'wb') as handle:
            handle.write(buffer)
        manifest_stages.append({
            'id': stage_id, 'file': rel, 'packedByteLength': len(buffer),
            'sha256': hashlib.sha256(bytes(buffer)).hexdigest(),
        })

    with open(shape_path) as handle:
        shape_text = handle.read()
    with open(os.path.join(out_dir, 'shape.json'), 'w') as handle:
        handle.write(shape_text)

    meta = {
        'kind': 'student',
        'shape': shape_name,
        'shapeFile': 'shape.json',
        'step': int(step),
        'params': int(sum(int(np.prod(t['shape'])) for t in tensors)),
        'blendScale': float(model.blend_scale.detach().cpu().reshape(-1)[0]),
        'version': model_version(shape_name, step),
    }
    if extra:
        meta.update(extra)
    manifest = {
        'totals': {'blockCount': len(block_idx), 'tensorCount': len(tensors),
                   'paramCount': meta['params']},
        'model': meta,
        'stages': manifest_stages,
        'tensors': tensors,
    }
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as handle:
        json.dump(manifest, handle, indent=1)
    return manifest


def is_student_package(directory):
    path = os.path.join(directory, 'manifest.json')
    if not os.path.isfile(path):
        return False
    with open(path) as handle:
        manifest = json.load(handle)
    return manifest.get('model', {}).get('kind') == 'student'


def read_package(directory, verify=True):
    """→ (OrderedDict name->torch.Tensor f32, manifest);默认逐 stage 校验 sha256。"""
    with open(os.path.join(directory, 'manifest.json')) as handle:
        manifest = json.load(handle)
    stages = {}
    for entry in manifest['stages']:
        path = os.path.join(directory, 'model', entry['file'])
        with open(path, 'rb') as handle:
            data = handle.read()
        if verify:
            digest = hashlib.sha256(data).hexdigest()
            if digest != entry['sha256']:
                raise ValueError(f'sha256 mismatch for {entry["file"]}: {digest} != {entry["sha256"]}')
        stages[entry['id']] = data

    state = OrderedDict()
    for entry in manifest['tensors']:
        blob = stages[entry['stage']][entry['stageOffset']:entry['stageOffset'] + entry['byteLength']]
        arr = np.frombuffer(blob, dtype='<f4').reshape(entry['shape']).copy()
        expected = int(np.prod(entry['shape'])) * 4
        if entry['byteLength'] != expected:
            raise ValueError(f"{entry['name']}: byteLength {entry['byteLength']} != {expected}")
        state[entry['name']] = torch.from_numpy(arr)
    return state, manifest


def load_student_package(directory, width, height, device):
    """包 → 可前向的 StudentNetwork(形状从包内 shape.json 自动探测)。"""
    state, manifest = read_package(directory)
    shape_file = manifest['model'].get('shapeFile', 'shape.json')
    shape = load_student_shape(os.path.join(directory, shape_file))
    g = geometry_from_valid(width, height)
    torch.manual_seed(0)
    model = StudentNetwork(shape, g).to(device).eval()
    model.load_state_dict(state)
    return model, manifest
