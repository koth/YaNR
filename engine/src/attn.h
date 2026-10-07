// 注意力数值核心(M3):把 cosine 归一 / prior 打分 / 查询维 softmax / 加权
// 求和融成单个 (window, head) 核心。输入是 gather 后的 [slots][96] 连续行
// (q/k/v 各 32;一次 memcpy 一整行,热点连续 —— 直接散行访问会把 k 行
// 甩出 L1,dots 反复重读反而更慢)。q/k 就地归一(gather 缓冲私有,安全)。
// 语义与学生 torch 逐点一致:
//   - 掩码在查询维:非法查询行整行丢弃;
//   - 非法键不掩码 —— 带 prior 分吸概率、值为 0(实现里跳过其点积/求和,等价);
//   - ViT 的 padded 槽在 softmax 中置 0;
//   - ViT 的 k 归一只做一次(原实现逐查询重复归一同一原始 k,确定性计算,
//     提前一次逐位同值)。
#pragma once

// qkv: [slots][row_stride] gather 缓冲(可写;row_stride = heads*96,含各 head
// 切片,调用方传本 head 偏移);槽 s 的输出行 = out + index[s]*out_stride
// (valid[s]=0 的槽不访问、不写);prior/scores: [slots][slots]。
void attn_window_head(float* qkv, int row_stride, const int* index,
                      const unsigned char* valid, int slots, float scale,
                      const float* prior, float* scores, float* out, int out_stride);

// qkv: [tokens][row_stride] gather 缓冲(可写);scores: [tokens][padded];
// out: token ti 写到 out + ti*out_stride;scale 含 sqrt(d) 因子(调用方乘好)。
void attn_vit_head(float* qkv, int row_stride, int tokens, int padded, float scale,
                   float* scores, float* out, int out_stride);

// 核心内部微剖析(norms/scores/softmax/av 累计毫秒;多线程粗累加,只看占比)。
const double* attn_profile();
void attn_profile_reset();
