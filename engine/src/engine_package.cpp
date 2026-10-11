#include "engine.h"
#include "json_read.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;
namespace {
// Package file names are relative to their declared root.
fs::path relative_file(const fs::path& root, const std::string& name) {
    auto p = fs::u8path(name);
    if (p.empty() || p.is_absolute() || p.has_root_path()) throw std::runtime_error("invalid package path: " + name);
    for (const auto& part : p) if (part == "..") throw std::runtime_error("invalid package path: " + name);
    return root / p;
}
}

Engine::Engine(const std::string& package_dir, int width, int height) {
    if (width < 33 || height < 33 || width > 8192 || height > 8192)
        throw std::runtime_error("image dimensions must be between 33 and 8192");
    const auto root = fs::u8path(package_dir);
    const auto manifest = jsonread::read(root / "manifest.json");
    const auto& model = manifest.at("model");
    if (model.at("kind").str() != "student") throw std::runtime_error("expected a student package");
    const auto shape = jsonread::read(relative_file(root, model.at("shapeFile").str()));
    struct Stage { size_t offset, bytes; };
    std::map<std::string, Stage> stages;
    for (const auto& entry : manifest.at("stages").list()) {
        auto path = relative_file(root / "model", entry.at("file").str());
        size_t bytes = static_cast<size_t>(entry.at("packedByteLength").integer());
        if (bytes == 0 || bytes % sizeof(float) || bytes != fs::file_size(path))
            throw std::runtime_error("stage size mismatch: " + path.u8string());
        size_t offset = bin_.size();
        if (!stages.emplace(entry.at("id").str(), Stage{offset, bytes}).second)
            throw std::runtime_error("duplicate stage id");
        std::ifstream file(path, std::ios::binary);
        bin_.resize(offset + bytes);
        if (!file.read(bin_.data() + offset, static_cast<std::streamsize>(bytes)))
            throw std::runtime_error("cannot read " + path.u8string());
    }
    // Populate pointers only after all stages have been read (vector reallocation).
    for (const auto& entry : manifest.at("tensors").list()) {
        const auto& name = entry.at("name").str();
        if (entry.at("dtype").str() != "f32") throw std::runtime_error("unsupported dtype: " + name);
        auto stage = stages.find(entry.at("stage").str());
        if (stage == stages.end()) throw std::runtime_error("unknown tensor stage: " + name);
        size_t offset = static_cast<size_t>(entry.at("stageOffset").integer());
        size_t bytes = static_cast<size_t>(entry.at("byteLength").integer());
        if (offset % sizeof(float) || offset > stage->second.bytes || bytes > stage->second.bytes - offset)
            throw std::runtime_error("tensor outside stage: " + name);
        Tensor tensor;
        size_t expected = sizeof(float);
        for (const auto& d : entry.at("shape").list()) {
            int dim = d.positive();
            if (expected > bytes / static_cast<size_t>(dim)) throw std::runtime_error("invalid tensor shape: " + name);
            expected *= dim;
            tensor.shape.push_back(dim);
        }
        if (expected != bytes) throw std::runtime_error("tensor size mismatch: " + name);
        tensor.data = reinterpret_cast<const float*>(bin_.data() + stage->second.offset + offset);
        if (!tensors_.emplace(name, tensor).second) throw std::runtime_error("duplicate tensor: " + name);
    }
    geom_ = geometry_from_valid(width, height);
    valid_size_ = width;
    vit_tokens_ = geom_.vit_tokens;
    vit_padded_ = geom_.padded_vit_tokens;
    if (geom_.levels[0].width % 8 || geom_.levels[0].height % 8)
        throw std::runtime_error("dimensions produce an unsupported d0 grid (must be divisible by 8)");
    const auto& blend = t("blend_scale");
    if (blend.shape != std::vector<int>{1}) throw std::runtime_error("invalid blend_scale");
    blend_scale_ = blend.data[0];

    static const char* levels[] = {"full", "d0", "d1", "d2", "d3", "d4"};
    static const int phases[4][2] = {{0, 0}, {4, 4}, {4, 0}, {0, 4}};
    std::map<std::string, jsonread::Value> configs;
    for (const auto& lv : shape.at("levels").list()) {
        if (!configs.emplace(lv.at("level").str(), lv).second) throw std::runtime_error("duplicate shape level");
    }
    for (const auto* name : levels) {
        const auto& lv = configs.at(name);
        BlockCfg cfg;
        cfg.kind = lv.at("kind").str();
        int channels = lv.at("channels").positive();
        cfg.hidden = lv.at("hidden").positive();
        cfg.attn = lv.has("attn") ? lv.at("attn").flag() : true;
        cfg.window = cfg.attn ? (lv.has("window_slots") ? lv.at("window_slots").positive() : 8) : 0;
        if (cfg.kind == "expert") { cfg.experts = channels / 32; cfg.narrow = 32; }
        else if (cfg.kind == "moe") {
            cfg.experts = lv.at("experts").positive();
            cfg.narrow = channels;
            if (cfg.experts < 2) throw std::runtime_error("moe needs at least 2 experts");
        }
        else if (cfg.kind == "split") {
            cfg.branches = lv.at("branches").positive();
            cfg.bc = lv.at("branch_channels").positive();
            cfg.mc = lv.at("middle_channels").positive();
        } else if (cfg.kind != "ffn") throw std::runtime_error("unsupported block kind: " + cfg.kind);
        if (channels % 32 || (cfg.kind == "split" && cfg.branches * cfg.bc != channels))
            throw std::runtime_error("invalid block channels");
        int count = lv.at("blocks").positive();
        int enc = std::max(1, count / 2), dec = std::max(1, count - enc);
        for (int side = 0; side < 2; ++side) {
            int n = side == 0 ? enc : dec;
            for (int j = 0; j < n; ++j) {
                int phase = (side == 0 ? j : enc + j) & 3;
                cfg.shift_x = cfg.attn ? phases[phase][0] : 0;
                cfg.shift_y = cfg.attn ? phases[phase][1] : 0;
                blocks_[side][name].push_back(cfg);
            }
        }
    }
    const auto& vit = shape.at("vit");
    if (vit.at("tokens_level").str() != "d5") throw std::runtime_error("unsupported ViT token level");
    BlockCfg cfg;
    cfg.kind = vit.has("experts") ? "moe_vit" : "vit";
    cfg.attn = true; cfg.hidden = vit.at("ffn").positive();
    cfg.experts = vit.has("experts") ? vit.at("experts").positive() : 0;
    vit_blocks_.assign(vit.at("blocks").positive(), cfg);
    pack_weights();
}
