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

Engine::Engine(const std::string& package_dir, int width, int height, bool cpu_pack) {
    if (width < 33 || height < 33 || width > 8192 || height > 8192)
        throw std::runtime_error("image dimensions must be between 33 and 8192");
    const auto root = fs::u8path(package_dir);
    const auto manifest = jsonread::read(root / "manifest.json");
    const auto& model = manifest.at("model");
    if (model.at("kind").str() != "student") throw std::runtime_error("expected a student package");
    const auto shape = jsonread::read(relative_file(root, model.at("shapeFile").str()));
    const std::string architecture = shape.has("architecture") ? shape.at("architecture").str() : "";
    cnn_ = architecture == "cnn_unet_moe8";
    const bool shared_routed = shared_routed_ = architecture == "shared-routed-moe-v1";
    if (!architecture.empty() && !cnn_ && !shared_routed)
        throw std::runtime_error("unsupported architecture: " + architecture);
    if (shared_routed) {
        if (shape.at("architecture").str() != "shared-routed-moe-v1" ||
            model.at("architecture").str() != "shared-routed-moe-v1" ||
            model.at("precision").str() != "fp32-tf32-off" || model.at("biasPolicy").str() != "frozen-inference")
            throw std::runtime_error("incompatible shared/routed package contract");
        const auto& routing = shape.at("routing");
        shared_experts_ = static_cast<int>(routing.at("shared_experts").integer());
        routed_experts_ = routing.at("routed_experts").positive();
        if ((shared_experts_ != 0 && shared_experts_ != 1) || routed_experts_ > 16)
            throw std::runtime_error("unsupported expert bank layout");
        const std::string layout = shared_experts_ ? "bank[shared,routed0..15,in,out]" :
            "bank[routed0.." + std::to_string(routed_experts_ - 1) + ",in,out]";
        // Older exporters used the shared-bank label for the routed-only bank.
        if (model.at("expertLayout").str() != layout &&
            !(shared_experts_ == 0 && model.at("expertLayout").str() == "bank[shared,routed0..15,in,out]"))
            throw std::runtime_error("incompatible expert layout");
        for (const auto* owner : {&shape, &model}) {
            const auto& r = owner->at("routing");
            if (r.at("fusion_scale").kind != jsonread::Value::Number || r.at("fusion_scale").number != 0.5 ||
                r.at("jitter").kind != jsonread::Value::Number || r.at("jitter").number != 0.0)
                throw std::runtime_error("shared/routed requires fusion_scale=0.5 and jitter=0");
            if (r.at("shared_experts").integer() != shared_experts_ || r.at("routed_experts").integer() != routed_experts_ || r.at("top_k").integer() != 1 ||
                r.at("spatial_granularity").str() != "window8" || r.at("vit_granularity").str() != "token" ||
                r.at("selection").str() != "fp32-argmax-with-bias-v1" || r.at("tie_break").str() != "lowest-index" ||
                r.at("gate").str() != "softmax-selected-times-routed-v1" ||
                r.at("fusion").str() != "sub-layer-mean-single-attention-v1" ||
                r.at("transitions").str() != "expertized")
                throw std::runtime_error("incompatible shared/routed routing semantics");
        }
    }
    // Early CNN exports declare the architecture only in shape.json. Tensor
    // dtypes and every CNN tensor shape are still validated below.
    if (cnn_ && ((model.has("architecture") && model.at("architecture").str() != architecture) ||
                 (model.has("precision") && model.at("precision").str() != "fp32-tf32-off")))
        throw std::runtime_error("incompatible CNN package contract");
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

    if (cnn_) {
        if (shape.at("experts").positive() != 8) throw std::runtime_error("CNN requires eight experts");
        static const char* names[] = {"full", "d0", "d1", "d2", "d3", "d4", "d5"};
        auto expect = [&](const std::string& key, std::vector<int> dims) {
            if (t(key).shape != dims) throw std::runtime_error("CNN tensor shape mismatch: " + key);
        };
        auto norm = [&](const std::string& p, int c) { expect(p + ".weight", {c}); expect(p + ".bias", {c}); };
        auto conv = [&](const std::string& p, int ci, int co, int k) {
            expect(p + ".weight", {co, ci, k, k}); expect(p + ".bias", {co});
        };
        for (int i = 0; i < 7; ++i) {
            const std::string name = names[i];
            int c = cnn_channels_[i] = shape.at("channels").at(name).positive();
            if (c % 8 || c > 1024) throw std::runtime_error("CNN channels must be a multiple of eight, at most 1024");
            int h = shape.at("expert_hidden").at(name).positive();
            if (i == 0) { conv("stem.0", 16, c, 3); norm("stem.1", c); }
            else { conv("enc_proj." + name + ".0", cnn_channels_[i - 1], c, 1); norm("enc_proj." + name + ".1", c); }
            for (int side = 0; side < (i == 6 ? 1 : 2); ++side) {
                int count = static_cast<int>(shape.at(side ? "blocks_dec" : "blocks_enc").at(name).integer());
                if (count > 128) throw std::runtime_error("too many CNN blocks");
                auto& bs = blocks_[side][name];
                for (int j = 0; j < count; ++j) {
                    BlockCfg cfg; cfg.kind = "cnn"; cfg.hidden = h; cfg.experts = 8; bs.push_back(cfg);
                    const std::string p = std::string(side ? "dec." : "enc.") + name + "." + std::to_string(j);
                    norm(p + ".norm1", c); norm(p + ".norm2", c); conv(p + ".conv", c, c, 3);
                    conv(p + ".ffn.router", c, 8, 1); expect(p + ".aux", {1, c, 1, 1});
                    for (int e = 0; e < 8; ++e) {
                        std::string b = p + ".ffn.bank." + std::to_string(e);
                        expect(b + ".0.weight", {h, c}); expect(b + ".0.bias", {h});
                        expect(b + ".2.weight", {c, h}); expect(b + ".2.bias", {c});
                    }
                }
            }
        }
        for (int i = 0; i < 6; ++i) {
            conv(std::string("merge.") + names[i] + ".0", cnn_channels_[i + 1] + cnn_channels_[i], cnn_channels_[i], 3);
            norm(std::string("merge.") + names[i] + ".1", cnn_channels_[i]);
        }
        norm("head_norm", cnn_channels_[0]); conv("head", cnn_channels_[0], 4, 1);
        prepare_cnn_weights();
        return;
    }

    static const char* levels[] = {"full", "d0", "d1", "d2", "d3", "d4"};
    static const int phases[4][2] = {{0, 0}, {4, 4}, {4, 0}, {0, 4}};
    std::map<std::string, jsonread::Value> configs;
    if (shared_routed && shape.at("levels").list().size() != 6)
        throw std::runtime_error("shared/routed requires exactly six spatial levels");
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
        cfg.dilation = lv.has("window_dilation") ? lv.at("window_dilation").positive() : 1;
        if (cfg.dilation > 8 || (cfg.dilation != 1 && !shared_routed))
            throw std::runtime_error("window dilation requires a routed student and must be at most eight");
        if (cfg.kind == "shared_routed_moe") {
            if (!shared_routed || channels > 1024 || cfg.hidden % 32 || (cfg.attn && cfg.window != 8) || lv.at("blocks").positive() < 2)
                throw std::runtime_error("invalid shared/routed spatial configuration");
            cfg.experts = shared_experts_ + routed_experts_; cfg.narrow = channels;
        }
        else if (shared_routed) throw std::runtime_error("shared/routed shape requires all layers expertized");
        else if (cfg.kind == "expert") { cfg.experts = channels / 32; cfg.narrow = 32; }
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
    cfg.kind = shared_routed ? "shared_routed_moe" : vit.has("experts") ? "moe_vit" : "vit";
    cfg.attn = true; cfg.hidden = vit.at("ffn").positive();
    cfg.experts = vit.has("experts") ? vit.at("experts").positive() : 0;
    if (shared_routed) {
        if (vit.at("kind").str() != "shared_routed_moe" || vit.at("channels").positive() % 32 || vit.at("channels").positive() > 1024 || cfg.hidden % 32)
            throw std::runtime_error("invalid shared/routed ViT configuration");
        cfg.experts = shared_experts_ + routed_experts_;
        const int E = cfg.experts, R = routed_experts_;
        auto expect = [&](const std::string& n, std::vector<int> dims) {
            if (t(n).shape != dims) throw std::runtime_error("shared/routed tensor shape mismatch: " + n);
        };
        auto router = [&](const std::string& p, int c) {
            expect(p + ".router.0.weight", {c}); expect(p + ".router.0.bias", {c});
            expect(p + ".router.1.weight", {R, c}); expect(p + ".router_bias", {R});
        };
        auto block = [&](const std::string& p, int c, int h, bool attn, bool global) {
            router(p, c); expect(p + ".expand", {E, c, h});
            expect(p + (global ? ".contract" : ".narrow"), {E, h, c}); expect(p + ".aux_ffn", {c});
            if (attn) {
                expect(p + ".qkv", {E, c, c * 3}); expect(p + ".proj", {E, c, c});
                expect(p + ".aux_attn", {c}); expect(p + ".attention.scale", {c / 32});
                if (!global) expect(p + ".attention.prior", {c / 32, 64, 64});
            }
        };
        auto transition = [&](const std::string& p, int ci, int co) {
            router(p, ci); expect(p + ".weight", {E, ci, co}); expect(p + ".bias", {E, co});
        };
        for (int i = 0; i < 6; ++i) {
            int c = configs.at(levels[i]).at("channels").positive();
            for (int side = 0; side < 2; ++side) {
                const auto& bs = blocks_[side].at(levels[i]);
                for (size_t j = 0; j < bs.size(); ++j)
                    block(std::string(side ? "dec_blocks." : "enc_blocks.") + levels[i] + "." + std::to_string(j), c, bs[j].hidden, bs[j].attn, false);
            }
            if (i > 0) transition(std::string("enc_trans.") + levels[i], configs.at(levels[i - 1]).at("channels").positive(), c);
            if (i > 0 && i < 5) transition(std::string("dec_trans.") + levels[i], configs.at(levels[i + 1]).at("channels").positive(), c);
        }
        int c = vit.at("channels").positive(), last = configs.at("d4").at("channels").positive();
        transition("vit_in", last, c); transition("vit_out", c, last);
        for (int j = 0; j < static_cast<int>(vit.at("blocks").integer()); ++j) block("vit_blocks." + std::to_string(j), c, cfg.hidden, true, true);
    }
    if (vit.at("blocks").integer() > 128) throw std::runtime_error("too many ViT blocks");
    vit_blocks_.assign(static_cast<size_t>(vit.at("blocks").integer()), cfg);
    if (cpu_pack) pack_weights();
}
