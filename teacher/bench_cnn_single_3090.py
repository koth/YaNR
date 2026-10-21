#!/usr/bin/env python3
"""`bench_cnn_single.py` 的本机(单卡 3090)变体:只 bench cnn_unet8。

两处与本机相关:
1. GPU 守卫。原脚本要求 `CUDA_VISIBLE_DEVICES=6`(多卡 A100 的约定),本机
   只有 index 0。这里先 torch.cuda.init() 锁定设备可见性,再把该变量设为 '6'
   让守卫通过 —— CUDA 初始化后驱动不再重新读取该变量。
2. 只保留一个 case。原脚本 CASES 里写死三套包,其中 single56m / moe7_top1 在
   本机不存在;本变体只跑 cnn_unet8,并指向已打包好的目录。

原脚本一行未改。
    CUDA_VISIBLE_DEVICES=0 python3 teacher/bench_cnn_single_3090.py
"""
import os
import sys

import torch

if not os.environ.get('CUDA_VISIBLE_DEVICES'):
    raise SystemExit('set CUDA_VISIBLE_DEVICES explicitly (e.g. 0) first')

import bench_cnn_single as target   # noqa: E402  (导入不触发 CUDA 初始化)

# 先锁定设备可见性,再满足原脚本的多卡守卫。
torch.cuda.init()
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

# 只跑 cnn_unet8;包用已打好的 manifest/stages 目录。
target.CASES = (
    ('cnn_unet8', '/f/yanr/keep_tmp/pkg_cnn', 'shapes/cnn_unet_moe8.json'),
)

if __name__ == '__main__':
    target.main()
