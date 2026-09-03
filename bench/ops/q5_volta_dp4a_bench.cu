// Direct microbench + correctness oracle for the int8/dp4a Q5 Volta GEMM (q5_volta_dp4a_gemm.cuh).
// Validates against launch_q5_simt_r8_c8 on the same randomised Q5 weight (codes + high plane +
// scales) so a wrong 5-bit decode / fragment map shows up as a mismatch, and times it against the
// fp16 mma route. RUN ON A V100 (CUDA_VISIBLE_DEVICES=1; titan's A2000 sits at PCI index 0).

#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include "ops/linear/q5/q5_launch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;

namespace {

void randomize_weight(bench::PackedQuantizedWeight& w, std::int32_t n, std::int32_t padded_k) {
    const std::size_t code_bytes = static_cast<std::size_t>(w.low_bytes);
    std::vector<std::uint8_t> codes(code_bytes);
    for (std::size_t i = 0; i < code_bytes; ++i) {
        codes[i] = static_cast<std::uint8_t>((i * 2654435761u + 1013904223u) >> 24);
    }
    w.storage.copy_from_host(codes.data(), code_bytes, 0);

    const std::size_t high_bytes = static_cast<std::size_t>(w.high_bytes);
    std::vector<std::uint8_t> high(high_bytes);
    for (std::size_t i = 0; i < high_bytes; ++i) {
        high[i] = static_cast<std::uint8_t>((i * 40503u + 12345u) >> 3);
    }
    w.storage.copy_from_host(high.data(), high_bytes, static_cast<std::size_t>(w.high_offset));

    const std::size_t groups = static_cast<std::size_t>(n) * (padded_k / 64);
    const std::uint16_t kScaleCycle[4] = {0x3c00, 0x3400, 0x3a00, 0x3800};
    std::vector<std::uint16_t> scales(groups);
    for (std::size_t g = 0; g < groups; ++g) { scales[g] = kScaleCycle[g & 3]; }
    w.storage.copy_from_host(scales.data(), scales.size() * sizeof(std::uint16_t),
                             static_cast<std::size_t>(w.scale_offset));
}

double mag_rel_error(const std::vector<std::uint16_t>& ref, const std::vector<std::uint16_t>& got) {
    auto bf16 = [](std::uint16_t b) {
        std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    };
    float mx = 1e-6f, we = 0.0f;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        mx = std::max(mx, std::abs(bf16(ref[i])));
        we = std::max(we, std::abs(bf16(ref[i]) - bf16(got[i])));
    }
    return static_cast<double>(we / mx);
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::int32_t n = 5120, k = 8192; // MLP down shape
        std::vector<std::int32_t> tokens{256, 512, 1024, 2048};
        for (int i = 1; i < argc; ++i) {
            std::string_view a(argv[i]);
            if (a == "--n") { n = std::stol(argv[++i]); }
            else if (a == "--k") { k = std::stol(argv[++i]); }
        }
        std::int32_t max_t = 0;
        for (auto t : tokens) { max_t = std::max(max_t, t); }

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        int dev = 0;
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        std::printf("device %d: %s (sm_%d%d)\n", dev, prop.name, prop.major, prop.minor);

        DeviceBuffer x = bench::make_bf16(static_cast<std::size_t>(k) * max_t);
        DeviceBuffer out_dp(static_cast<std::size_t>(n) * max_t * 2);
        DeviceBuffer out_simt(static_cast<std::size_t>(n) * max_t * 2);
        DeviceBuffer out_mma(static_cast<std::size_t>(n) * max_t * 2);
        bench::PackedQuantizedWeight weight = bench::make_row_split_weight(QType::Q5G64_F16S, n, k, k);
        randomize_weight(weight, n, k);
        CUDA_CHECK(cudaDeviceSynchronize());
        WorkspaceArena ws(static_cast<std::size_t>(n) * max_t * sizeof(float) + (256u << 20));

        std::printf("Q5 volta_dp4a GEMM  n=%d k=%d\n", n, k);
        for (const std::int32_t t : tokens) {
            Tensor xt(x.p, DType::BF16, {k, t});
            Tensor odp(out_dp.p, DType::BF16, {n, t});
            Tensor osimt(out_simt.p, DType::BF16, {n, t});
            Tensor omma(out_mma.p, DType::BF16, {n, t});

            detail::launch_q5_simt_r8_c8(xt, weight.weight, osimt, stream);
            detail::launch_q5_volta_dp4a(xt, weight.weight, odp, /*add_residual=*/false, 0, ws, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<std::uint16_t> h_ref(static_cast<std::size_t>(n) * t), h_got(h_ref.size());
            out_simt.copy_to_host(h_ref.data(), h_ref.size() * 2);
            out_dp.copy_to_host(h_got.data(), h_got.size() * 2);
            const double rel = mag_rel_error(h_ref, h_got);
            const double flops = 2.0 * n * k * t;

            const bench::ColdTiming td = bench::measure_launch(
                [&](cudaStream_t s) {
                    detail::launch_q5_volta_dp4a(xt, weight.weight, odp, false, 0, ws, s);
                }, stream, 5, 30);
            const bench::ColdTiming tm = bench::measure_launch(
                [&](cudaStream_t s) {
                    detail::launch_q5_volta_mma(xt, weight.weight, omma, false, 0, ws, s);
                }, stream, 5, 30);
            std::printf("  T=%-5d %-6s mag_rel_err=%.4f%%   dp4a %.2f TFLOP/s   mma %.2f TFLOP/s\n",
                        t, rel < 0.01 ? "PASS" : "FAIL", rel * 100.0,
                        flops / (td.median_us * 1e-6) / 1e12, flops / (tm.median_us * 1e-6) / 1e12);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "q5_volta_dp4a_bench failed: %s\n", e.what());
        return 1;
    }
}
