#pragma once
// Windows Imaging Component: PNG/JPEG/BMP/TIFF input without Python/OpenCV.
#include <filesystem>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <wincodec.h>
#include <wrl/client.h>
#endif

struct InputImage { int width = 0, height = 0; std::vector<float> rgb; };
inline InputImage read_image(const std::string& path, int width, int height) {
#ifdef _WIN32
    using Microsoft::WRL::ComPtr;
    auto check = [](HRESULT hr) {
        if (FAILED(hr)) throw std::runtime_error("image decode failed (WIC HRESULT " + std::to_string(hr) + ")");
    };
    struct ComScope {
        HRESULT result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        ~ComScope() { if (SUCCEEDED(result)) CoUninitialize(); }
    } com;
    if (com.result != RPC_E_CHANGED_MODE) check(com.result);
    ComPtr<IWICImagingFactory> factory;
    check(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)));
    ComPtr<IWICBitmapDecoder> decoder;
    check(factory->CreateDecoderFromFilename(std::filesystem::u8path(path).c_str(), nullptr, GENERIC_READ,
                                            WICDecodeMetadataCacheOnLoad, &decoder));
    ComPtr<IWICBitmapFrameDecode> frame;
    check(decoder->GetFrame(0, &frame));
    UINT source_w, source_h;
    check(frame->GetSize(&source_w, &source_h));
    if (width == 0) width = static_cast<int>(source_w);
    if (height == 0) height = static_cast<int>(source_h);
    if (width < 33 || height < 33 || width > 8192 || height > 8192)
        throw std::runtime_error("image dimensions must be between 33 and 8192");
    ComPtr<IWICBitmapSource> source;
    check(frame.As(&source));
    ComPtr<IWICBitmapScaler> scaler;
    if (width != static_cast<int>(source_w) || height != static_cast<int>(source_h)) {
        check(factory->CreateBitmapScaler(&scaler));
        check(scaler->Initialize(frame.Get(), width, height, WICBitmapInterpolationModeFant));
        check(scaler.As(&source));
    }
    ComPtr<IWICFormatConverter> converter;
    check(factory->CreateFormatConverter(&converter));
    check(converter->Initialize(source.Get(), GUID_WICPixelFormat24bppRGB, WICBitmapDitherTypeNone,
                                nullptr, 0, WICBitmapPaletteTypeCustom));
    std::vector<unsigned char> bytes(static_cast<size_t>(width) * height * 3);
    check(converter->CopyPixels(nullptr, width * 3, static_cast<UINT>(bytes.size()), bytes.data()));
    InputImage out;
    out.width = width; out.height = height; out.rgb.resize(bytes.size());
    for (size_t i = 0; i < bytes.size(); ++i) out.rgb[i] = bytes[i] / 255.0f;
    return out;
#else
    (void)path; (void)width; (void)height;
    throw std::runtime_error("--image currently requires Windows WIC; use --features on this platform");
#endif
}
