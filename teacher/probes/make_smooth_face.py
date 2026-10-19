"""造"游戏滑脸"探针输入:低通滤掉皮肤微纹理,保留大结构(对齐 portraits_smooth 口径)。

用法: python3 make_smooth_face.py in.png out.png [radius]
"""
import sys

from PIL import Image, ImageFilter

img = Image.open(sys.argv[1]).convert('RGB')
radius = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
img.filter(ImageFilter.GaussianBlur(radius=radius)).save(sys.argv[2])
print('wrote', sys.argv[2], 'radius', radius)
