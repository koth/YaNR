// cpu_engine: image inference and forward-only benchmark, plus the original parity CLI.
//   cpu_engine photo.jpg --weights weights/student_mixed --png photo_nr.png --bench
//   cpu_engine --idx student512.idx --features feat.bin --out head.bin
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

#include "engine.h"
#include "gemm.h"
#include "lanes.h"
#include "png_write.h"
#include "attn.h"
#include "image_read.h"
#ifdef NR_HAS_CUDA
#include "engine_cuda.h"
#endif

namespace fs = std::filesystem;
#ifdef NR_DEFAULT_CUDA
#define NR_CLI_NAME "gpu_engine"
#else
#define NR_CLI_NAME "cpu_engine"
#endif
namespace {
void usage() {
    std::puts(
        "usage: " NR_CLI_NAME " IMAGE [--weights weights/student_mixed] [--png OUTPUT.png]\n"
        "                  [--size N | --width W --height H] [--bench] [--repeats 30]\n"
        "                  [--warmup 5] [--threads 8] [--seed 12345] [--style 0..127]\n"
        "                  [--tone 0.5] [--structure 0.5] [--skin -1] [--automask]\n"
        "                  [--blend-png blend.png] [--proxy-png proxy.png]\n"
        "                  [--backend cpu|cuda] [--device 0]\n"
        "       " NR_CLI_NAME " --idx MODEL.idx --features INPUT.bin [--out HEAD.bin | --bench]\n"
        "                  [--proxy PROXY.bin --png OUTPUT.png] [--dump-dir DIR]\n"
        "Image size defaults to the input size. -o aliases --png; --out writes raw f32 head.\n"
        "Threads default to OMP_NUM_THREADS when set, otherwise 8.\n"
        "--style is the style ID/conditioning value (0 keeps the default; nominal range 0..127).\n"
        "The usable range depends on checkpoint training; student_mixed is intended for style 0..3.\n"
        "--tone/--structure control lanes directly; --skin is used only with --automask.\n"
        "With --features, lane options are ignored because the input lanes are prebuilt.\n"
        "--weights can also be combined with --features (default size 512x512).\n"
        "Benchmark times only the network forward after warmup, without image/lane/I/O work.\n"
        "NR_INT8=1 enables mixed int8; NR_VNNI=0 disables VNNI auto-dispatch;\n"
        "NR_PROFILE=1 enables diagnostic profiling.");
#ifdef NR_HAS_CUDA
    std::puts("CUDA is available: gpu_engine defaults to CUDA; cpu_engine defaults to CPU.\n"
              "CUDA benchmarks use resident inputs and CUDA events; transfers are reported separately.\n"
              "CUDA uses FP32 (TF32 disabled); NR_INT8/NR_PROFILE are CPU-only.");
#endif
}
int integer(const std::string& text, int low, int high) {
    size_t end;
    long long v = std::stoll(text, &end);
    if (end != text.size() || v < low || v > high) throw std::runtime_error("invalid integer: " + text);
    return static_cast<int>(v);
}
double real(const std::string& text) {
    size_t end;
    double v = std::stod(text, &end);
    if (end != text.size() || !std::isfinite(v)) throw std::runtime_error("invalid number: " + text);
    return v;
}
void mkdir_parent(const std::string& path) {
    auto parent = fs::u8path(path).parent_path();
    if (!parent.empty()) fs::create_directories(parent);
}
void read_f32(const std::string& path, std::vector<float>& data) {
    auto p = fs::u8path(path);
    if (fs::file_size(p) != data.size() * sizeof(float)) throw std::runtime_error("size mismatch: " + path);
    std::ifstream file(p, std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float)))
        throw std::runtime_error("cannot read " + path);
}
void write_f32(const std::string& path, const std::vector<float>& data) {
    mkdir_parent(path);
    std::ofstream file(fs::u8path(path), std::ios::binary);
    if (!file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float)))
        throw std::runtime_error("cannot write " + path);
    file.close();
    if (!file) throw std::runtime_error("cannot finish writing " + path);
}
void write_png(const std::string& path, int width, int height, const uint8_t* rgb) {
    mkdir_parent(path);
    if (!pngw::write_rgb8(path, width, height, rgb)) throw std::runtime_error("cannot write " + path);
    std::printf("wrote %s\n", path.c_str());
}
double elapsed(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
int run(const std::vector<std::string>& args) {
    std::string idx_path, weights_path, image_path, feat_path, out_path, dump_dir;
    std::string proxy_path, png_path, blend_path, proxy_png, dump_features;
    int repeats = 30, warmup = 5, threads = 0, width = 0, height = 0;
    bool bench = false;
    int device = 0;
#ifdef NR_DEFAULT_CUDA
    std::string backend = "cuda";
#else
    std::string backend = "cpu";
#endif
    LaneParams lanes;
    lanes.seed = 12345;
    for (size_t i = 1; i < args.size(); ++i) {
        const auto& a = args[i];
        auto next = [&]() -> const std::string& {
            if (++i == args.size()) throw std::runtime_error("missing value after " + a);
            return args[i];
        };
        if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (a == "--idx") idx_path = next();
        else if (a == "--weights") weights_path = next();
        else if (a == "--image") image_path = next();
        else if (a == "--features") feat_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--dump-dir") dump_dir = next();
        else if (a == "--dump-features") dump_features = next();
        else if (a == "--proxy") proxy_path = next();
        else if (a == "--png" || a == "-o") png_path = next();
        else if (a == "--blend-png") blend_path = next();
        else if (a == "--proxy-png") proxy_png = next();
        else if (a == "--bench") bench = true;
        else if (a == "--backend") backend = next();
        else if (a == "--device") device = integer(next(), 0, 255);
        else if (a == "--repeats") repeats = integer(next(), 1, 10000);
        else if (a == "--warmup") warmup = integer(next(), 1, 10000);
        else if (a == "--threads") threads = integer(next(), 1, 256);
        else if (a == "--size") width = height = integer(next(), 33, 8192);
        else if (a == "--width") width = integer(next(), 33, 8192);
        else if (a == "--height") height = integer(next(), 33, 8192);
        else if (a == "--seed") {
            auto seed = next(); size_t end;
            auto n = std::stoull(seed, &end);
            if (end != seed.size() || n > UINT32_MAX) throw std::runtime_error("invalid seed: " + seed);
            lanes.seed = static_cast<uint32_t>(n);
        }
        else if (a == "--style") {
            lanes.style = real(next());
            if (lanes.style < 0.0 || lanes.style > 127.0)
                throw std::runtime_error("--style must be between 0 and 127");
        }
        else if (a == "--tone") lanes.tone = real(next());
        else if (a == "--structure") lanes.structure = real(next());
        else if (a == "--skin") lanes.skin = real(next());
        else if (a == "--automask") lanes.auto_mask = true;
        else if (!a.empty() && a[0] != '-' && image_path.empty()) image_path = a;
        else throw std::runtime_error("unknown argument: " + a);
    }
    if (args.size() == 1) { usage(); return 2; }
    if (backend != "cpu" && backend != "cuda") throw std::runtime_error("--backend must be cpu or cuda");
    const bool use_cuda = backend == "cuda";
#ifndef NR_HAS_CUDA
    if (use_cuda) throw std::runtime_error("CUDA was not built; configure with -DNR_CUDA=ON");
#endif
    if (use_cuda && (std::getenv("NR_INT8") || std::getenv("NR_PROFILE")))
        throw std::runtime_error("NR_INT8/NR_PROFILE are CPU-only; unset them for CUDA inference");
    if ((!image_path.empty()) == (!feat_path.empty())) throw std::runtime_error("specify one of IMAGE or --features");
    if (!idx_path.empty() && !weights_path.empty()) throw std::runtime_error("specify one of --idx or --weights");
    if (bench && !dump_dir.empty()) throw std::runtime_error("--dump-dir copies tensors; run it separately from --bench");
    if (!image_path.empty() && !proxy_path.empty()) throw std::runtime_error("--image already provides the proxy");
    if (!idx_path.empty() && (width || height)) throw std::runtime_error("--idx fixes the geometry; use --weights to resize");
    if (weights_path.empty() && idx_path.empty()) weights_path = "weights/student_mixed";
    if (!threads) threads = std::getenv("OMP_NUM_THREADS") ? omp_get_max_threads() : 8;
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    std::unique_ptr<Engine> engine;
    auto t0 = std::chrono::steady_clock::now();
    if (!idx_path.empty()) engine = std::make_unique<Engine>(idx_path, !use_cuda);
    double load_ms = elapsed(t0);
    if (engine && !image_path.empty()) {
        width = engine->geometry().valid_width; height = engine->geometry().valid_height;
    }
    std::vector<float> proxy;
    double image_ms = 0;
    if (!image_path.empty()) {
        t0 = std::chrono::steady_clock::now();
        auto image = read_image(image_path, width, height);
        width = image.width; height = image.height; proxy = std::move(image.rgb);
        image_ms = elapsed(t0);
        if (png_path.empty()) {
            auto p = fs::u8path(image_path);
            png_path = (p.parent_path() / fs::u8path(p.stem().u8string() + "_nr.png")).u8string();
        }
    }
    if (!engine) {
        if (!width) width = 512;
        if (!height) height = 512;
        t0 = std::chrono::steady_clock::now();
        engine = std::make_unique<Engine>(weights_path, width, height, !use_cuda);
        load_ms = elapsed(t0);
    }
    const Geometry& g = engine->geometry();
    width = g.valid_width; height = g.valid_height;
    if (image_path.empty() && !proxy_path.empty()) {
        proxy.resize(static_cast<size_t>(width) * height * 3);
        read_f32(proxy_path, proxy);
    }
    if (proxy.empty() && (!png_path.empty() || !blend_path.empty() || !proxy_png.empty()))
        throw std::runtime_error("PNG output needs IMAGE or --proxy");
    if (!bench && out_path.empty() && png_path.empty()) throw std::runtime_error("specify --out or --png");
    std::vector<float> features(static_cast<size_t>(g.full_rows) * 16);
    t0 = std::chrono::steady_clock::now();
    if (!image_path.empty()) build_lanes(proxy.data(), width, height, g, lanes, features.data());
    else read_f32(feat_path, features);
    double features_ms = elapsed(t0);
    if (!dump_features.empty()) write_f32(dump_features, features);
    std::vector<float> head(static_cast<size_t>(g.full_rows) * 4);
    const bool profile = std::getenv("NR_PROFILE") != nullptr;
    engine->set_debug(!dump_dir.empty());
    engine->set_profile(profile);
    engine->set_int8(std::getenv("NR_INT8") != nullptr);
#ifdef NR_HAS_CUDA
    std::unique_ptr<CudaEngine> gpu;
    double gpu_load_ms = 0, upload_ms = 0, download_ms = 0;
    if (use_cuda) {
        t0 = std::chrono::steady_clock::now();
        gpu = std::make_unique<CudaEngine>(*engine, device);
        gpu_load_ms = elapsed(t0);
        t0 = std::chrono::steady_clock::now();
        gpu->upload(features.data());
        upload_ms = elapsed(t0);
        std::printf("CUDA device %d: %s  FP32 (TF32 off), resident weights/features, CUDA Graph\n",
                    device, gpu->device_name().c_str());
    }
#endif
    auto forward = [&]() {
#ifdef NR_HAS_CUDA
        if (gpu) return gpu->forward(!dump_dir.empty());
#endif
        auto start = std::chrono::steady_clock::now();
        engine->forward(features.data(), head.data());
        return elapsed(start);
    };
    std::printf("model %s  valid %dx%d  field %dx%d  threads %d  style %.3g tone %.3g structure %.3g skin %.3g automask %d  mode %s\n",
                idx_path.empty() ? weights_path.c_str() : idx_path.c_str(), width, height,
                g.full_width, g.full_height, omp_get_max_threads(), lanes.style, lanes.tone,
                lanes.structure, lanes.skin, lanes.auto_mask ? 1 : 0,
                use_cuda ? "cuda-fp32" : (std::getenv("NR_INT8") ? (std::string("mixed-int8/") + gemm_i8_backend()).c_str() : "fp32"));

    if (bench) {
        for (int i = 0; i < warmup; ++i) forward();
        engine->set_profile(profile);
        // Only enable attention's internal timers for explicit diagnostic profiling.
        if (profile && std::getenv("NR_ATTN_PROFILE")) attn_profile_reset();
        std::vector<double> times;
        times.reserve(repeats);
        for (int i = 0; i < repeats; ++i) {
            times.push_back(forward());
        }
        std::sort(times.begin(), times.end());
        double median = (times[(repeats - 1) / 2] + times[repeats / 2]) * 0.5;
        double p90 = times[static_cast<size_t>(std::ceil(repeats * 0.9)) - 1];
        std::printf("forward only: median %.3f ms  min %.3f ms  p90 %.3f ms  (%d warmup, %d repeats)\n",
                    median, times.front(), p90, warmup, repeats);
        if (profile) {
            const double* p = engine->profile();
            double inv = 1.0 / repeats;
            std::printf("profile: gemm %.2f  silu %.2f  attn %.2f (gather %.2f)  other %.2f ms\n",
                        p[0] * inv, p[1] * inv, p[2] * inv, p[3] * inv,
                        median - (p[0] + p[1] + p[2]) * inv);
            if (std::getenv("NR_ATTN_PROFILE")) {
                const double* ap = attn_profile();
                std::printf("attn-core accumulated: norms %.2f  scores %.2f  softmax %.2f  av %.2f ms\n",
                            ap[0] * inv, ap[1] * inv, ap[2] * inv, ap[3] * inv);
            }
            std::vector<std::pair<double, std::string>> layers;
            for (const auto& kv : engine->layer_profile()) layers.emplace_back(kv.second * inv, kv.first);
            std::sort(layers.rbegin(), layers.rend());
            std::puts("top layers (block: timings include their child operations):");
            for (size_t i = 0; i < std::min<size_t>(30, layers.size()); ++i)
                std::printf("  %7.3f ms  %s\n", layers[i].first, layers[i].second.c_str());
        }
    } else {
        t0 = std::chrono::steady_clock::now();
        forward();
        std::printf("forward (first call, includes scratch/cache initialization): %.3f ms\n", elapsed(t0));
    }
#ifdef NR_HAS_CUDA
    if (gpu) {
        t0 = std::chrono::steady_clock::now();
        gpu->download(head.data());
        download_ms = elapsed(t0);
        std::printf("outside forward: CUDA init+weight upload %.2f ms  H2D features %.2f ms  D2H head %.2f ms\n",
                    gpu_load_ms, upload_ms, download_ms);
        for (const auto& kv : gpu->debug())
            write_f32((fs::u8path(dump_dir) / (kv.first + ".bin")).u8string(), kv.second);
    }
#endif
    if (!out_path.empty()) write_f32(out_path, head);
    for (const auto& kv : engine->debug())
        write_f32((fs::u8path(dump_dir) / (kv.first + ".bin")).u8string(), kv.second);
    std::printf("outside forward: weights%s %.2f ms  image %.2f ms  features %.2f ms\n",
                use_cuda ? " load" : "+packing", load_ms, image_ms, features_ms);

    // First-frame composite: neural = clamp(proxy + head_rgb/4). The fourth
    // channel is the temporal history blend; without history it does not mix proxy back in.
    if (!proxy.empty()) {
        t0 = std::chrono::steady_clock::now();
        std::vector<uint8_t> rgb(proxy.size()), bw(proxy.size());
        double rmin = 1e30, rmax = -1e30, rsum = 0, lmin = 1e30, lmax = -1e30, bsum = 0;
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            const float* hv = head.data() + (static_cast<size_t>(y) * g.full_width + x) * 4;
            size_t offset = (static_cast<size_t>(y) * width + x) * 3;
            for (int c = 0; c < 3; ++c) {
                double r = hv[c] / 4.0;
                double v = std::clamp(proxy[offset + c] + r, 0.0, 1.0);
                rgb[offset + c] = static_cast<uint8_t>(v * 255.0 + 0.5);
                rmin = std::min(rmin, r); rmax = std::max(rmax, r); rsum += r;
            }
            double lg = hv[3];
            double bl = std::clamp(1.0 / (1.0 + std::exp(-lg)) * engine->blend_scale(), 0.0, 1.0);
            for (int c = 0; c < 3; ++c) bw[offset + c] = static_cast<uint8_t>(bl * 255.0 + 0.5);
            lmin = std::min(lmin, lg); lmax = std::max(lmax, lg); bsum += bl;
        }
        std::printf("composite %.3f ms (outside forward): rgb/4 min %+.3f max %+.3f mean %+.3f | "
                    "blend logit min %+.3f max %+.3f | blend mean %.3f\n",
                    elapsed(t0), rmin, rmax, rsum / proxy.size(), lmin, lmax,
                    bsum / (static_cast<double>(width) * height));
        if (!png_path.empty()) write_png(png_path, width, height, rgb.data());
        if (!blend_path.empty()) write_png(blend_path, width, height, bw.data());
        if (!proxy_png.empty()) {
            for (size_t i = 0; i < proxy.size(); ++i) rgb[i] = static_cast<uint8_t>(proxy[i] * 255.0 + 0.5);
            write_png(proxy_png, width, height, rgb.data());
        }
    }
    return 0;
}
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    try {
        // Keep benchmark output visible when launched with redirected Windows pipes.
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        std::vector<std::string> args;
        for (int i = 0; i < argc; ++i) {
#ifdef _WIN32
            args.push_back(fs::path(argv[i]).u8string());
#else
            args.emplace_back(argv[i]);
#endif
        }
        return run(args);
    } catch (const std::exception& e) {
        std::fprintf(stderr, NR_CLI_NAME ": %s\n", e.what());
        return 1;
    }
}
