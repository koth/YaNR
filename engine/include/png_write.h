#pragma once
// 极简 PNG 编码器(stored-deflate,零依赖,openspec 8.4):8-bit RGB。
// 产物是合法 PNG(不压缩,文件偏大;离线验收/对比足够)。
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace pngw {

inline uint32_t crc32(const uint8_t* d, size_t n) {
    static uint32_t tab[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            tab[i] = c;
        }
        init = true;
    }
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) crc = tab[(crc ^ d[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

inline uint32_t adler32(const uint8_t* d, size_t n) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < n; i++) {
        a = (a + d[i]) % 65521;
        b = (b + a) % 65521;
    }
    return (b << 16) | a;
}

inline void put32be(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((uint8_t)(x >> 24)); v.push_back((uint8_t)(x >> 16));
    v.push_back((uint8_t)(x >> 8));  v.push_back((uint8_t)x);
}

inline void chunk(std::vector<uint8_t>& out, const char* type,
                  const std::vector<uint8_t>& data) {
    put32be(out, (uint32_t)data.size());
    size_t crc_start = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data.begin(), data.end());
    put32be(out, crc32(out.data() + crc_start, out.size() - crc_start));
}

// rgb: w*h*3 字节(行主序)。返回 false = 打不开文件。
inline bool write_rgb8(const std::string& path, int w, int h, const uint8_t* rgb) {
    std::vector<uint8_t> raw;
    raw.reserve((size_t)h * ((size_t)w * 3 + 1));
    for (int y = 0; y < h; y++) {
        raw.push_back(0);                              // filter: none
        raw.insert(raw.end(), rgb + (size_t)y * w * 3, rgb + (size_t)(y + 1) * w * 3);
    }
    std::vector<uint8_t> z = {0x78, 0x01};             // zlib 头(无压缩 hint)
    size_t off = 0;
    while (off < raw.size()) {
        size_t n = std::min<size_t>(65535, raw.size() - off);
        z.push_back(off + n >= raw.size() ? 1 : 0);    // BFINAL + BTYPE=00(stored)
        z.push_back((uint8_t)(n & 0xFF)); z.push_back((uint8_t)(n >> 8));
        z.push_back((uint8_t)(~n & 0xFF)); z.push_back((uint8_t)((~n >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + off, raw.begin() + off + n);
        off += n;
    }
    put32be(z, adler32(raw.data(), raw.size()));

    std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<uint8_t> ihdr;
    put32be(ihdr, (uint32_t)w);
    put32be(ihdr, (uint32_t)h);
    ihdr.push_back(8);    // bit depth
    ihdr.push_back(2);    // color type: truecolor RGB
    ihdr.push_back(0); ihdr.push_back(0); ihdr.push_back(0);
    chunk(out, "IHDR", ihdr);
    chunk(out, "IDAT", z);
    chunk(out, "IEND", {});
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);
    return true;
}

}  // namespace pngw
