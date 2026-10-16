#!/usr/bin/env bash
# Run t1..t7 at native size; image decode uses Pillow, all NR work uses C++.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
weights="$repo_dir/weights/student_v43"
samples_dir="$repo_dir/teacher/samples"
engine=''
lanes=''
python_bin="${NR_SAMPLES_PYTHON:-}"
style=2 tone=1 structure=1 skin=-1 seed=12345 threads=8 device=0
mask=--automask
samples=()
usage() {
    cat <<'EOF'
Usage: bash engine/run_samples.sh [options] [t1 t2 ...]
Defaults: t1..t7, weights/student_v43, native size, style=2 tone=1 structure=1
          skin=-1 automask=1 seed=12345 threads=8; overwrite nr.png, no backups.
Options: --weights DIR --samples-dir DIR --engine FILE --lanes FILE --python FILE
         --style N --tone N --structure N --skin N --seed N --threads N --device N
         --no-automask --help
EOF
}
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --no-automask) mask=--no-automask; shift ;;
        --weights|--samples-dir|--engine|--lanes|--python|--style|--tone|--structure|--skin|--seed|--threads|--device)
            if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                --weights) weights=$2 ;; --samples-dir) samples_dir=$2 ;;
                --engine) engine=$2 ;; --lanes) lanes=$2 ;; --python) python_bin=$2 ;;
                --style) style=$2 ;; --tone) tone=$2 ;; --structure) structure=$2 ;;
                --skin) skin=$2 ;; --seed) seed=$2 ;; --threads) threads=$2 ;; --device) device=$2 ;;
            esac
            shift 2 ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) samples+=("$1"); shift ;;
    esac
done
if ((${#samples[@]} == 0)); then samples=(t1 t2 t3 t4 t5 t6 t7); fi
if [[ -z "$engine" ]]; then
    for candidate in "$script_dir/build-cuda/gpu_engine" "$script_dir/build/cpu_engine"; do
        if [[ -x "$candidate" ]]; then engine=$candidate; break; fi
    done
fi
if [[ -z "$engine" || ! -x "$engine" ]]; then
    echo 'No engine found. Build the C++ engine first or pass --engine FILE.' >&2; exit 1
fi
if [[ -z "$lanes" ]]; then lanes="$(dirname -- "$engine")/cpu_lanes"; fi
if [[ ! -x "$lanes" ]]; then echo "Missing cpu_lanes: $lanes; build it or pass --lanes FILE." >&2; exit 1; fi
if [[ ! -f "$weights/manifest.json" ]]; then echo "Missing student package: $weights" >&2; exit 1; fi
if [[ -z "$python_bin" ]]; then
    if [[ -x "$repo_dir/tmp/samples-venv/bin/python" ]]; then python_bin="$repo_dir/tmp/samples-venv/bin/python"
    elif [[ -x "$repo_dir/.venv/bin/python" ]]; then python_bin="$repo_dir/.venv/bin/python"
    else python_bin=python3; fi
fi
if ! "$python_bin" -c 'from PIL import Image' >/dev/null 2>&1; then
    echo 'Pillow is required for image decoding. Set up from the repository root:' >&2
    echo '  uv venv tmp/samples-venv --python 3.10' >&2
    echo '  uv pip install --python tmp/samples-venv/bin/python pillow' >&2
    exit 1
fi
# Preflight every original before overwriting any existing outputs.
sources=()
shopt -s nullglob
for sample in "${samples[@]}"; do
    if [[ ! "$sample" =~ ^t[1-9][0-9]*$ ]]; then echo "Invalid sample name: $sample" >&2; exit 2; fi
    matches=()
    for path in "$samples_dir/$sample/$sample".*; do
        case "${path,,}" in *.png|*.jpg|*.jpeg|*.bmp|*.tif|*.tiff|*.webp) matches+=("$path") ;; esac
    done
    if ((${#matches[@]} != 1)); then echo "Expected one original image for $sample; found ${#matches[@]}." >&2; exit 1; fi
    sources+=("${matches[0]}")
done
temp_root="$(cd -- "${TMPDIR:-/tmp}" && pwd -P)"
work_dir="$(mktemp -d "$temp_root/yanr-samples.XXXXXXXX")"
cleanup() {
    case "$work_dir" in "$temp_root"/yanr-samples.*) rm -rf -- "$work_dir" ;; esac
}
trap cleanup EXIT
export OMP_NUM_THREADS="$threads"
unset NR_INT8 NR_PROFILE
controls=(--style "$style" --tone "$tone" --structure "$structure" --skin "$skin" --seed "$seed" "$mask")
echo "Engine: $engine"
for source in "${sources[@]}"; do
    dimensions="$("$python_bin" "$script_dir/tools/decode_proxy.py" "$source" "$work_dir/proxy.f32")"
    read -r width height <<< "$dimensions"
    output="$(dirname -- "$source")/nr.png"
    echo "== $source: ${width}x${height} -> $output =="
    "$lanes" --proxy "$work_dir/proxy.f32" --out "$work_dir/features.f32" \
        --vw "$width" --vh "$height" "${controls[@]}"
    "$engine" --weights "$weights" --features "$work_dir/features.f32" \
        --proxy "$work_dir/proxy.f32" --width "$width" --height "$height" \
        "${controls[@]}" --threads "$threads" --device "$device" -o "$output"
done
echo "Done: ${#sources[@]} images; nr.png overwritten without backups."
