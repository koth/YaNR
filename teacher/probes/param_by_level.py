import sys, torch
sys.path.insert(0, '/home/koth/klss/teacher')
from nr_geometry import geometry_from_valid
from nr_student import StudentNetwork, load_student_shape

g = geometry_from_valid(320, 512)
for tag, path in (('v4.4', '/home/koth/klss/shapes/student_moe_v0.json'),
                  ('v4.5', '/home/koth/klss/shapes/student_shared_moe_v1.json')):
    torch.manual_seed(0)
    m = StudentNetwork(load_student_shape(path), g)
    per = {}
    for name, p in m.named_parameters():
        parts = name.split('.')
        key = parts[1] if len(parts) > 1 else name
        if key in ('enc_blocks', 'dec_blocks'):
            key = f'{key}.{parts[2]}'
        elif key in ('enc_trans', 'dec_trans', 'merge'):
            key = f'{key}.{parts[2]}'
        elif parts[0] == 'vit_blocks':
            key = 'vit_blocks'
        per[key] = per.get(key, 0) + p.numel()
    total = sum(per.values())
    print(f'== {tag}  总参数 {total:,} ==')
    for k in sorted(per, key=lambda x: (x.split(".")[-1], x)):
        if per[k] > 1000:
            print(f'   {k:<22}{per[k]:>12,}  {100*per[k]/total:5.1f}%')
