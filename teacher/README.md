# teacher — the OpenDLSS-NR network, self-contained

The teacher implementation for the fast-student distillation project: a faithful, bit-exact port of the
OpenDLSS-NR neural rendering network (71 blocks) to torch + native CUDA, able to run the network on a CUDA
GPU without WebGPU or Vulkan — including GPUs without FP8 tensor cores (written for a 3090, Ampere sm_86).

This directory is **self-contained**: the oracle, the three backends, the native kernels, the validation
fixtures and even the C++ reference sources needed by the strongest tie-breaker all live here. It was
extracted from `OpenDLSS-NR/ports/torch` (plus `ports/browser-webgpu/web/fixtures/numerics.bin` and
`src/reference.*`); nothing outside this directory is required except the model weights and a CUDA GPU.

"Faithful" means the arithmetic follows the same publication grid as the reference: `docs/numerics.md`,
`src/reference.cpp` and `numerics.js` define it, and every step here is checked against the recorded
fixture `fixtures/numerics.bin` before anything runs.

## Layout

| path | what it is |
| --- | --- |
| `nr_numerics.py` | the scalar oracle — a line-for-line port of the reference numerics |
| `nr_torch.py` | the vectorized kernels (fixed-point chains, publications) + oracle checks |
| `nr_triton.py` | the fused Triton kernels (fallback backend) |
| `nr_kernels.cu` / `nr_cuda.py` | the native CUDA kernels (GEMM, f16 GEMM, fused window attention, fused block FFN, fused ViT attention), driven by ctypes |
| `nr_geometry.py` | padded field, six levels, window phases, block byte layouts |
| `nr_model.py` | manifest/stage loading, MMA-fragment weight unpacking |
| `nr_network.py` | the 71 blocks (elementwise, blocks, expert FFN, window attention, ViT) |
| `run_nr.py` | the runner: bench input, timings, head statistics, dumps for parity |
| `profile_nr.py` / `bench_detail.py` | stage timings / per-shape, per-kernel breakdowns |
| `check_numerics.py` | oracle + all backends against `fixtures/numerics.bin` |
| `check_model.py` | weight unpacking against the reference's scalar addressing |
| `check_fused.py` | the fused kernels (window, block FFN, expert FFN, ViT) against the step-by-step paths |
| `check_block0.py` / `check_block0.cpp` | the C++ tie-breaker, built against the vendored `reference.cpp` |
| `compare_heads.py` / `compare_bnd.py` | bitwise comparison of head / boundary dumps |
| `fixtures/numerics.bin` | the recorded C++ reference for every numerics case |
| `vendor/reference/` | `reference.cpp`, `reference.h`, `numeric.h` — the executable arbiter |
| `shim/nr_model.h` | the model handle the reference's loader expects |

The model weights are external data: a directory with `manifest.json` + stage tensors (the synthetic
teacher is at `~/klss/models/nr-synth` on the dev server), passed as `NR_WEIGHTS`.

## Quick start

```sh
# 1. the numerics must pass first (pins every backend to the oracle and the fixture)
python3 check_numerics.py --torch

# 2. the weight loader against the reference addressing
python3 check_model.py $NR_WEIGHTS

# 3. the fused kernels against the step-by-step paths
python3 check_fused.py

# 4. the network on the GPU (the .so builds itself on first use)
NR_WEIGHTS=~/klss/models/nr-synth python3 run_nr.py --width 512 --height 512 --frames 5

# 5. the C++ tie-breaker (optional; needs a host compiler)
g++ -O2 -std=c++17 -I shim check_block0.cpp -o check_block0
NR_WEIGHTS=~/klss/models/nr-synth python3 check_block0.py /tmp/parity
```

`run_nr.py` uses a deterministic input (a smooth field in lanes 0-3, xorshift noise in lanes 4-6, seed
12345), so its output is comparable value-for-value across ports and across student versions.

## The cost model, measured (3090)

This is what the student's architecture must respect — everything was measured on this implementation:

* **Frame time ≈ MACs ÷ ~600 G products/s.** The fixed-point chain is the wall; memory is not (the
  intermediate tensors move at ~50 GB/s, ~7% of the card). Activation tensor size does not matter much;
  the multiply count does.
* For a dense layer, **MACs per pixel = that layer's parameter count** (every weight multiplies every
  pixel). Attention is different: MACs ∝ width × key count. So "same parameters, smaller activations"
  buys nothing on the FFN layers — the fast-student levers are **activated parameters** (MoE/conditional
  compute), **width²** (channels ÷ √5 ≈ parameters ÷ 5 per layer), or **spatial sparsity**.
* Where the teacher's 512×512 frame (~69 G MACs) goes: full-resolution + level-0 blocks (ch=32, 10 blocks)
  28%; the wide deep stages (ch=256/512, 1472 pixels total!) 31%; the ViT (8 × 1024×4096 FFN over 96
  tokens) 14%; levels 64/128 (expert FFN) 23%; window attention everywhere 16%.
* Measured kernel throughput (the ceilings a student can assume): fused block chains 892 G products/s,
  GEMMs 530–710, window attention ~630. Reference frame times, bit-exact: 320² 57 ms, 512² 113 ms,
  768² 231 ms, 1080p 720 ms.

## Using this as the teacher

* `run_nr.py --dump DIR` writes `features.f32.bin` (the input) and `head.f32.bin` (f32 [rows][4]) — the
  distillation target. `network.boundaries` captures block outputs for feature matching (`--dump` also
  writes them under `bnd/`).
* `compare_heads.py` is the acceptance test for a student: bit-exact against the teacher is the goal on
  the fixture paths; on real content, track the residual statistics `run_nr.py` prints.
* The student can reuse this runtime verbatim: the kernels are shape-generic (strided, tile size chosen
  per launch), so a student with different channel counts needs only its own `nr_geometry`-style layout
  and `nr_network`-style graph. `NR_TRITON=0` selects plain torch loops; `NR_FUSED_PREP=0`/
  `NR_FUSED_BLOCK=0`/`NR_FUSED_EXPERT=0`/`NR_FUSED_VIT=0` bisect the fused kernels.

## The contract, kept literally

* f16 publications round-to-nearest-even; E4M3 is the FP8 publication (`fp8_quant`), NaN → +0.
* FP8 GEMM: K in two 16-product groups per 32; per group a shared exponent, each product aligned to 13
  fractional bits and truncated toward zero, exact integer sum, one rounding to half; the group result is
  the next group's accumulator (`adaFp8Fdpa16`).
* f16 GEMM (input adapter, head): 8-product groups, 24 fractional bits, exact integer → half
  (`adaF16Fdpa8` / `fixedToF16`).
* K partitions (ViT contract 1024, qkv 512, projection 256) accumulate independently and combine with half
  adds — `gemmFp8Element`'s `partitionSums`.
* The residual is the accumulator seed scaled by the block's learned per-channel half vector, **not** a
  post-add.
* Window attention: cosine normalization folded in (pair squares, half tree, 1/sqrt in f32), the learned
  prior as the score GEMM's accumulator, `expWeight` in the half exponent field, the 64-wide half-add
  softmax tree over 4×4-tiled key order, the value fold in 4×16 groups, `fp8Domain` publications.
* The global ViT: `sqrt(32)` and the learned scale as separate half multiplies, `vitExpWeight`,
  unnormalized weights with the reciprocal through the value sum, and the `padding * exp(0)` denominator
  correction.
* Elementwise: the 2×2 pool `((a+b)+(c+d)) * 0.25` in halves, `upsample_residual`'s single rounding of
  `in + skip * scale`, `post_blend`'s two-step blend.
* The A operand of every FP8 GEMM is rotated within each 32-wide K block (`packedInputIndex`), applied at
  load; batch offsets are tensor strides (broadcast = stride 0).

## Kernels

The backends run native CUDA first (`nr_kernels.cu`, standalone C ABI built with nvcc and driven through
ctypes — no torch extension, no version coupling), then Triton (`nr_triton.py`), then plain torch loops.
Each backend carries the whole chain — the group loop, the shared exponent, the per-term truncations, the
half rounding after every 16 products — plus the epilogue (the A-operand swizzle, the residual ×
per-channel-scale seed, MpCubicSiLU, the E4M3 publication, the raw half) in one launch. Beyond that the
native path fuses harder, all of it bit-identical:

* the FP8 GEMM tiles 32×32 outputs with 2×2 thread tiles (one shared-memory operand per product, padded
  rows so nothing conflicts), and runs the ViT's partitioned chains (independent chunks per K partition,
  half-add combined) inside the same launch — `partition` is a kernel argument;
* the chain's per-group exponent scan is shared across a 2×2 thread tile (`fp8_group_quad`): a row of
  operands feeds two outputs, so one scan pass covers four chains — about +35% on the GEMMs;
* the window attention starts from the **raw** qkv: the cosine normalization and the q/k/v publications
  (about ten full-tensor torch passes) happen inside the kernel, one block per (window, head) over all 64
  queries, with the published k staged transposed so the score phase reads without bank conflicts;
* the 32-channel block's FFN chain (expand 32→128 with SiLU → contract 128→32 with the residual seed →
  qkv 32→96) is one launch per block with the intermediates never leaving shared memory
  (`nr_cuda.block_ffn`); the expert blocks get the same treatment (`nr_cuda.block_ffn_expert`: E broadcast
  expands → E narrows → contract → qkv, streamed expert by expert), dispatched only when the row grid is
  dense enough (below ~16·ch rows the batched gemms' wider grids win);
* the ViT's normalize + attend is one kernel per (head, 8 queries) from the raw qkv
  (`nr_cuda.vit_attention_raw`): the cosine norm, the three-rounding q publication (norm, √32, learned
  scale), the vitExpWeight scores, the chunked 64-wide softmax trees with the running half adds and the
  padding correction, and the value chain (accumulator kept in registers across chunks).

Every backend is checked bit-for-bit against the fixture and the scalar oracle (`check_numerics.py --torch`
runs all of them; `check_fused.py` pins the fused kernels against the step-by-step paths).

## Real weights

The teacher runs NVIDIA's actual model once the weights are extracted from `nvngx_dlssnr.dll`
(the network of DLSS 5; the 310.8.x builds carry the same 71-block graph):

```sh
# 1. extract the packed WEIGHTS_HT resource (MLX-DLSS's tooling, Apache 2.0)
git clone https://github.com/iamwavecut/MLX-DLSS tools/MLX-DLSS
uv venv --python 3.13 tools/mlxdlss-venv && uv pip install --python tools/mlxdlss-venv/bin/python tools/MLX-DLSS/python
tools/mlxdlss-venv/bin/mlxdlss-weights all nvngx_dlssnr.dll weights/extracted

# 2. pack it into a model directory this runtime loads
python3 convert_weights.py weights/extracted/dlssnr-weights-packed.safetensors -o weights/nr

# 3. run it
NR_WEIGHTS=weights/nr python3 run_image.py samples/lake.png --width 512 --height 512 -o out
```

Extracted models retain NVIDIA's terms and must not be redistributed.

Validation chain on the real model: `check_model.py` (loader), `check_block0.py` (the port vs the compiled
`vendor/reference/reference.cpp`, **0/32 mismatched on real weights**), `check_fused.py`, plus three
cross-implementation tools against MLX-DLSS's independent recovery: `check_vs_mlx.py` (decoded matrices),
`check_vs_mlx_head.py` (head-to-head on identical features), `check_mlx_reweight.py` (their model with
re-ordered weights). The two recoveries agree only partially (head correlation 0.5-0.65): their decode
carries per-block-family swizzles fitted on vendor stage captures, and their pipeline pads some sizes
differently (e.g. 512² to a 512² field where the documented geometry is 576×512). Our chain is bit-exact
against the vendored C++ reference; final arbitration between the two recoveries needs NVIDIA-side
captures (the `parity` fixtures, which are not distributed).

The `NR_WEIGHT_LAYOUT` switch (default `nr`) selects the E4M3 fragment mapping; `mlx` selects MLX-DLSS's
for the cross-checks.

## Notes

* A frame is a fixed sequence of launches, so `run_nr.py` captures it into a CUDA graph after warm-up and
  replays it per frame — no Python or launch cost per op (`--eager` runs frame by frame instead).
* The plain-torch reference chains loop over k-groups sequentially — the accumulator chain after every 16
  products is the contract, so the groups cannot be reordered or summed in parallel.
* `MAX_GROUP_ELEMENTS` in `nr_torch.py` bounds the loop path's group-step intermediates (128 MiB of f32 at
  a time).
* E4M3 tensors are carried as `torch.float16` values on the E4M3 grid — every E4M3 value is exactly a
  half, so the encode/decode tables are only used at the publications.
* `check_block0.cpp` includes the vendored `vendor/reference/reference.cpp` directly; build it with
  `-I shim` (the shim supplies `nr_model.h`).

## The distilled student (fast-student-distillation)

A small model distilled from the teacher's head contract: same 16-lane input (Box-Muller noise ×3, the
centred proxy, the reprojected previous output, the conditioning lanes), same composite
(`clamp(proxy + rgb/4)` + `clamp(sigmoid(logit) · blend_scale)`), **one deterministic feed-forward
network — not a diffusion model**. Structure, acceptance numbers and the full rationale live in
[`STUDENT.md`](STUDENT.md) and [`report_final.md`](report_final.md).

### Student structure (shipped shape: `shapes/student_slim_mixed.json`)

| level | channels | hidden | blocks (enc+dec) | kind | attention |
| --- | --- | --- | --- | --- | --- |
| full | 32 | 32 | 1+1 | FFN | none |
| d0 | 32 | 32 | 3+3 | FFN | 8×8 window |
| d1 | 64 | 64 | 3+3 | expert (E=2) | 8×8 window |
| d2 | 128 | 64 | 2+2 | expert (E=2) | 8×8 window |
| d3 | 192 | 64 | 2+2 | expert (E=2) | 8×8 window |
| d4 | 192 | 64 (branch 32, middle 128) | 2+2 | split (6 branches) | 8×8 window |
| ViT | 384 | FFN 768 | 2 | ViT (d5 tokens) | global |

**4,286,790 parameters** (fp32 17.2 MB / f16 8.6 MB); **3.34 GMAC @320² / 9.32 GMAC @512²**.
Alternate shapes in `shapes/`: `student_v0` (13.71 GMAC, the quality baseline), `student_slim_blocks`,
`student_slim_deep`, `student_alt_depth`, `student_alt_attn4x4`.

### Cost model / shape sweeps

```sh
python3 cost_model.py --shape ../shapes/student_slim_mixed.json --sizes 320,512,768,1080 [--json]
python3 sweep_shapes.py          # shape scanning (see STUDENT.md §7 for the accounting)
```

### Training / evaluation / packaging

```sh
# distillation (online teacher; 60K steps ≈ 20h on a 3090)
python3 train_distill.py --teacher-weights $NR_WEIGHTS \
  --data ../data/raw/DIV2K_train_HR ../data/raw/DAVIS/JPEGImages/Full-Resolution ../data/train/proxy_proc \
  --shape ../shapes/student_slim_mixed.json --size 320,512 --size-weights 0.7,0.3 \
  --batch 4 --steps 60000 --warmup 4000 --pair-fraction 0.3 --loss-temporal 0.5 -o runs/v11

# acceptance (6.1-6.3): quality vs teacher, temporal stability
python3 eval_quality.py  --teacher-weights $NR_WEIGHTS --shape ../shapes/student_slim_mixed.json \
  --sizes 320,512 --checkpoint runs/v11/ckpt.pt --images '...' -o eval_v11
python3 eval_temporal.py --teacher-weights $NR_WEIGHTS --shape ../shapes/student_slim_mixed.json \
  --seq '.../*.jpg' --checkpoint runs/v11/ckpt.pt --size 512 --max-frames 4 -o eval_v11_t
#   --hist-src teacher = open-loop diagnostic (isolates the feedback loop)

# figures (6.5) and the C++ CPU engine parity + bench (8.4/8.8; NR_INT8=1, NR_PROFILE=1)
python3 quad_compare.py --image ... --size 512 --shape ../shapes/student_slim_mixed.json \
  --checkpoint runs/v11/ckpt.pt --teacher-weights $NR_WEIGHTS -o samples
python3 check_engine.py --image ... --size 512 --shape ../shapes/student_slim_mixed.json \
  --checkpoint runs/v11/ckpt.pt --bench

# package (7.1) + bit-exact round-trip check (7.3)
python3 convert_weights.py --shape student runs/v11/ckpt.pt -o weights/student_mixed
python3 check_package.py --package weights/student_mixed --checkpoint runs/v11/ckpt.pt

# run a packaged student (7.2): the shape is auto-detected from the manifest
# (the PNG outputs carry the model version in their tEXt metadata)
python3 run_image.py samples/lake.png --width 512 --height 512 \
  --weights weights/student_mixed --shape student -o out
python3 run_nr.py --width 512 --height 512 --frames 5 --weights weights/student_mixed
```

The student package has the teacher's shape (`manifest.json` + `model/stages/s*.bin`, per-stage sha256
and per-tensor stage offsets); the tensors are raw f32 and the shape DSL is embedded as `shape.json`, so
loading needs no external files (`nr_package.load_student_package`).

### Hardware requirements

* **Training**: a CUDA GPU; 60K steps ≈ 20h on a 3090. Dataset: DIV2K + DAVIS frames + the preprocessed
  proxy corpus (see `train_distill.py --help`).
* **torch runtime** (evaluation / rendering): CUDA or CPU; needs torch, numpy, Pillow.
* **CPU engine** (`../engine`): x86-64 AVX2 + OpenMP (CMake; MSVC `/O2 /arch:AVX2` supported). Eight
  threads measured optimal on an i9-10850K (512² end-to-end 91.5 ms, 320² 28.4 ms, fp32). The int8 path
  (`NR_INT8=1`, quality-identical) only pays on VNNI-class hardware — that is the 512²/30ms acceptance
  target (task 8.7).

### Provenance and licensing (发布声明)

* **The student weights are distillation artifacts**: they are trained against this port's teacher
  outputs and contain **no NVIDIA weights**. They may be redistributed on their own terms.
* **The teacher weights and the DLL are NVIDIA proprietary and are not redistributed with this source.**
  `weights/nr` (extracted from `nvngx_dlssnr.dll`) and the DLL itself stay under NVIDIA's terms; every
  user extracts them from their own copy of the DLL (see "Real weights" above).
* No NVIDIA weight data, no extracted stage files and no `nvngx_dlssnr.dll` may be committed to this
  repository or attached to a release of the student.
