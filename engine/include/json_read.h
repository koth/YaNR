#pragma once
// Small, read-only JSON parser for the student manifest/shape. No build-time download.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace jsonread {
struct Value {
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    bool boolean = false;
    double number = 0;
    std::string string;
    std::vector<Value> array;
    std::map<std::string, Value> object;
    const Value& at(const std::string& key) const {
        auto it = object.find(key);
        if (kind != Object || it == object.end()) throw std::runtime_error("missing JSON key: " + key);
        return it->second;
    }
    bool has(const std::string& key) const { return kind == Object && object.count(key); }
    const std::string& str() const {
        if (kind != String) throw std::runtime_error("expected JSON string");
        return string;
    }
    int64_t integer() const {
        if (kind != Number || !std::isfinite(number) || number < 0 ||
            number > 9007199254740991.0 || std::floor(number) != number)
            throw std::runtime_error("expected nonnegative JSON integer");
        return static_cast<int64_t>(number);
    }
    int positive() const {
        auto n = integer();
        if (n == 0 || n > std::numeric_limits<int>::max()) throw std::runtime_error("invalid JSON dimension");
        return static_cast<int>(n);
    }
    bool flag() const {
        if (kind != Bool) throw std::runtime_error("expected JSON boolean");
        return boolean;
    }
    const std::vector<Value>& list() const {
        if (kind != Array) throw std::runtime_error("expected JSON array");
        return array;
    }
};

class Parser {
    const std::string& s;
    size_t p = 0;
    [[noreturn]] void fail() const { throw std::runtime_error("invalid JSON at byte " + std::to_string(p)); }
    void space() { while (p < s.size() && (s[p] == ' ' || s[p] == '\r' || s[p] == '\n' || s[p] == '\t')) ++p; }
    bool take(char c) { space(); if (p < s.size() && s[p] == c) { ++p; return true; } return false; }
    uint32_t hex4() {
        uint32_t u = 0;
        for (int i = 0; i < 4; ++i) {
            if (p == s.size()) fail();
            char c = s[p++];
            int v = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 :
                    c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
            if (v < 0) fail();
            u = (u << 4) | v;
        }
        return u;
    }
    std::string text() {
        if (!take('"')) fail();
        std::string out;
        while (p < s.size()) {
            unsigned char c = s[p++];
            if (c == '"') return out;
            if (c < 32) fail();
            if (c != '\\') { out += c; continue; }
            if (p == s.size()) fail();
            c = s[p++];
            if (c == '"' || c == '\\' || c == '/') out += c;
            else if (c == 'b') out += '\b';
            else if (c == 'f') out += '\f';
            else if (c == 'n') out += '\n';
            else if (c == 'r') out += '\r';
            else if (c == 't') out += '\t';
            else if (c == 'u') {
                uint32_t u = hex4();
                if (u >= 0xD800 && u <= 0xDBFF) {
                    if (p + 2 > s.size() || s[p++] != '\\' || s[p++] != 'u') fail();
                    uint32_t low = hex4();
                    if (low < 0xDC00 || low > 0xDFFF) fail();
                    u = 0x10000 + ((u - 0xD800) << 10) + low - 0xDC00;
                } else if (u >= 0xDC00 && u <= 0xDFFF) fail();
                if (u < 0x80) out += static_cast<char>(u);
                else {
                    if (u >= 0x10000) out += static_cast<char>(0xF0 | (u >> 18));
                    if (u >= 0x800) out += static_cast<char>((u >= 0x10000 ? 0x80 : 0xE0) | ((u >> 12) & 0x3F));
                    out += static_cast<char>((u >= 0x800 ? 0x80 : 0xC0) | ((u >> 6) & 0x3F));
                    out += static_cast<char>(0x80 | (u & 0x3F));
                }
            } else fail();
        }
        fail();
    }
    Value value(int depth) {
        if (depth > 64) fail();
        space();
        if (p == s.size()) fail();
        Value v;
        if (s[p] == '"') { v.kind = Value::String; v.string = text(); }
        else if (take('{')) {
            v.kind = Value::Object;
            if (take('}')) return v;
            do {
                auto key = text();
                if (!take(':')) fail();
                if (!v.object.emplace(key, value(depth + 1)).second) fail();
            } while (take(','));
            if (!take('}')) fail();
        } else if (take('[')) {
            v.kind = Value::Array;
            if (take(']')) return v;
            do { v.array.push_back(value(depth + 1)); } while (take(','));
            if (!take(']')) fail();
        } else if (s.compare(p, 4, "null") == 0) { p += 4; }
        else if (s.compare(p, 4, "true") == 0) { p += 4; v.kind = Value::Bool; v.boolean = true; }
        else if (s.compare(p, 5, "false") == 0) { p += 5; v.kind = Value::Bool; }
        else {
            size_t start = p;
            if (s[p] == '-') ++p;
            if (p == s.size()) fail();
            if (s[p] == '0') ++p;
            else { if (s[p] < '1' || s[p] > '9') fail(); while (p < s.size() && s[p] >= '0' && s[p] <= '9') ++p; }
            if (p < s.size() && s[p] == '.') {
                ++p; size_t digits = p;
                while (p < s.size() && s[p] >= '0' && s[p] <= '9') ++p;
                if (p == digits) fail();
            }
            if (p < s.size() && (s[p] == 'e' || s[p] == 'E')) {
                ++p; if (p < s.size() && (s[p] == '+' || s[p] == '-')) ++p;
                size_t digits = p;
                while (p < s.size() && s[p] >= '0' && s[p] <= '9') ++p;
                if (p == digits) fail();
            }
            v.kind = Value::Number; v.number = std::stod(s.substr(start, p - start));
            if (!std::isfinite(v.number)) fail();
        }
        return v;
    }
public:
    explicit Parser(const std::string& input) : s(input) {}
    Value parse() { Value v = value(0); space(); if (p != s.size()) fail(); return v; }
};
inline Value read(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.u8string());
    std::string data{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    return Parser(data).parse();
}
}  // namespace jsonread
