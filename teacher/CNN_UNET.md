# CNN U-Net MoE8 student

独立于现有 Swin/ViT student 的新架构，文件和训练入口都不与
`nr_student.py` / `train_distill.py` 冲突。

## 文件

| path | role |
| --- | --- |
| `nr_cnn_unet.py` | 模型、8 专家路由、捕获点 |
| `train_distill_cnn.py` | 独立训练入口 |
| `cnn_unet_count.py` | 参数量/前向检查 |
| `../shapes/cnn_unet_moe8.json` | 结构规格 |
| `../remote/run_cnn_unet.json` | 服务器训练配置 |
| `../remote/start_train_cnn.sh` | 服务器启动脚本 |

## 结构

- 7 级 CNN U-Net：`full, d0..d5`
- 通道：`32, 32, 64, 128, 256, 256, 512`
- 编码块数：`2, 2, 3, 3, 3, 3, 2`
- 解码块数：`d4..full = 3, 3, 3, 3, 3, 2`
- 每块：共享 3x3 卷积残差 + 8 专家 top-1 FFN
- 专家：1x1 `Linear(ch -> hidden) -> SiLU -> Linear(hidden -> ch)`，
  按像素 argmax 选择，选中专家输出乘 softmax gate
- router 和最终 head 保持 FP32；其余学生矩阵可走 TF32
- 输入/输出契约与教师一致：输入 16 lane，输出 head `[rows, 4]`

## 参数量

实测：**62,116,253 params（62.116M）**

```text
stem        4,704
enc        39,379,216
enc_proj      244,384
merge       3,550,464
dec        18,937,288
head_norm         64
head             132
blend_scale        1
```

## 训练

```sh
cd code/teacher
source ../env/bin/activate
CUDA_VISIBLE_DEVICES=6 python3 train_distill_cnn.py \
  --config ../config/run_cnn_unet.json \
  --data ../data/full/DIV2K_train_HR ../data/full/Flickr2K \
         ../data/full/Full-Resolution ../data/full/proxy_proc \
         ../data/full/rgb ../data/full/gameplay \
         ../data/full/portraits ../data/full/frames_2k \
  -o ../runs/cnn_unet_moe8_50k
```

服务器直接运行：

```sh
bash /data2/hezm/klss/scripts/start_train_cnn.sh
```

## 当前 run

- 数据：完整 30,582 张
- 尺寸：320:512 = 7:3
- batch 10，50K steps，FP32 + TF32
- `moe_aux=0.01`，`moe_z=0.001`
- run：`/data2/hezm/klss/runs/cnn_unet_moe8_50k_run2`
- Comet：`https://www.comet.com/koth-chen/dlss-cnn-unet/cd30835b53664bdfb417b276946cca1e`
