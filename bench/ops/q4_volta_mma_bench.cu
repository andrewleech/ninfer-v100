// Direct microbench + correctness oracle for the fused-dequant Q4 Volta tensor-core GEMM
// (q4_volta_mma_gemm.cuh), the wide-T prefill kernel of the dual-device graph-shard MLP.
//
// The single-card public LinearSwiGLU Op does NOT route through this kernel -- it dequantises to
// global fp16 and runs CUTLASS -- so ninfer_q4_linear_swiglu_bench never exercises it. This binary
// calls launch_q4_volta_mma directly with a synthetic RowSplit Q4 weight and validates it against
// launch_q4_simt_r8_c8 (the trusted SIMT fallback) on the SAME randomized weight + activations, so
// a wrong tensor-core fragment map shows up as a mismatch rather than a silent miscompute.
//
//   ncu --kernel-name regex:q4_volta_mma_gemm_kernel ... ./ninfer_q4_volta_mma_bench --t-sweep 256
//
// RUN IT ON A V100. titan enumerates the RTX A2000 (sm_86) at PCI index 0, so with
// CUDA_DEVICE_ORDER=PCI_BUS_ID the default device 0 is the A2000, and this sm_70 binary JITs its
// compute_70 PTX forward onto it -- correct but ~12x slower, which silently masqueraded as a
// terrible kernel. Pin a V100 with CUDA_VISIBLE_DEVICES=1 (or 2); the startup line below prints the
// device it actually ran on. Measured on a V100-SXM2-16GB @ locked 1530MHz, kDirect single-split
// path (matches production shards): gate/up n=16384 k=5120 ~21-22 TFLOP/s, down n=5120 k=8192
// ~24-25 TFLOP/s across T=256..2048 -- ~20% of the ~112 TFLOP/s fp16/fp32-acc peak. ncu: L1/TEX
// 84.6%% (top), Compute 46%%, DRAM 25%%, IPC 1.88 -- L1/shared-bound (the shared weight staging),
// not compute- or issue-bound. Two attempted levers both lost: hoisting the bf16->fp16 activation
// convert out of the loop = +1%%; register-resident weights (drop the shared buffer, decode each
// lane's mirrored B row) = -83%% (goes DRAM-bound at 68%% -- the shared staging is what coalesces
// the scattered per-row weight loads). The kernel is near its practical optimum for this algorithm.

#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include "ops/linear/q4/q4_launch.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;

namespace {

struct Options {
    std::int32_t n = 4096; // output rows (matches the kernel's own documented N=4096 K=5120 point)
    std::int32_t k = 5120; // reduction dim (whole Q4 groups: k % 64 == 0)
    std::vector<std::int32_t> tokens{16, 32, 64, 128, 256, 512, 1024, 2048};
    int warmup    = 5;
    int repeat    = 30;
    bool validate = true;
};

std::vector<std::int32_t> parse_tokens(std::string_view raw) {
    std::vector<std::int32_t> out;
    std::size_t begin = 0;
    while (begin < raw.size()) {
        const std::size_t end = raw.find(',', begin);
        out.push_back(static_cast<std::int32_t>(
            std::stol(std::string(raw.substr(begin, end == std::string_view::npos ? raw.size() - begin
                                                                                  : end - begin)))));
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    return out;
}

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a(argv[i]);
        const auto next = [&](const char* what) -> std::string_view {
            if (i + 1 >= argc) { throw std::invalid_argument(std::string(what) + " needs a value"); }
            return argv[++i];
        };
        if (a == "--t-sweep") {
            o.tokens = parse_tokens(next("--t-sweep"));
        } else if (a == "--n") {
            o.n = static_cast<std::int32_t>(std::stol(std::string(next("--n"))));
        } else if (a == "--k") {
            o.k = static_cast<std::int32_t>(std::stol(std::string(next("--k"))));
        } else if (a == "--warmup") {
            o.warmup = std::stoi(std::string(next("--warmup")));
        } else if (a == "--repeat") {
            o.repeat = std::stoi(std::string(next("--repeat")));
        } else if (a == "--no-validate") {
            o.validate = false;
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(a));
        }
    }
    return o;
}

// Overwrite the constant fill the helper lays down with a varied pattern, so the SIMT-vs-MMA
// comparison actually exercises distinct weight rows and k-positions. Codes: a cheap hash per byte.
// Scales: cycle a few exact fp16 magnitudes near 1.0 per group so a dropped/mis-indexed scale
// changes the result.
void randomize_weight(bench::PackedQuantizedWeight& w, std::int32_t n, std::int32_t padded_k) {
    const std::size_t code_bytes = static_cast<std::size_t>(w.low_bytes);
    std::vector<std::uint8_t> codes(code_bytes);
    for (std::size_t i = 0; i < code_bytes; ++i) {
        codes[i] = static_cast<std::uint8_t>((i * 2654435761u + 1013904223u) >> 24);
    }
    w.storage.copy_from_host(codes.data(), code_bytes, 0);

    const std::size_t groups = static_cast<std::size_t>(n) * (padded_k / 64);
    const std::uint16_t kScaleCycle[4] = {0x3c00, 0x3400, 0x3a00, 0x3800}; // 1.0, 0.25, 0.75, 0.5
    std::vector<std::uint16_t> scales(groups);
    for (std::size_t g = 0; g < groups; ++g) { scales[g] = kScaleCycle[g & 3]; }
    w.storage.copy_from_host(scales.data(), scales.size() * sizeof(std::uint16_t),
                             static_cast<std::size_t>(w.scale_offset));
}

double max_rel_error(const std::vector<std::uint16_t>& ref, const std::vector<std::uint16_t>& got,
                     bool dump = false) {
    auto bf16 = [](std::uint16_t b) {
        std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    };
    double worst      = 0.0;
    std::size_t worst_i = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const float a = bf16(ref[i]);
        const float b = bf16(got[i]);
        const float denom = std::max(1e-3f, std::abs(a));
        const double r    = static_cast<double>(std::abs(a - b) / denom);
        if (r > worst) { worst = r; worst_i = i; }
    }
    (void)worst_i;
    (void)dump;
    return worst;
}

// Error normalised by the reference magnitude scale (max |ref|), the right metric for an int8
// path: near-zero outputs (from cancellation) carry large per-element relative error even when the
// absolute error is a tiny fraction of a typical output.
double magnitude_rel_error(const std::vector<std::uint16_t>& ref,
                           const std::vector<std::uint16_t>& got) {
    auto bf16 = [](std::uint16_t b) {
        std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    };
    float max_abs   = 1e-6f;
    float worst_err = 0.0f;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        max_abs   = std::max(max_abs, std::abs(bf16(ref[i])));
        worst_err = std::max(worst_err, std::abs(bf16(ref[i]) - bf16(got[i])));
    }
    return static_cast<double>(worst_err / max_abs);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse_options(argc, argv);
        const std::int32_t n = o.n;
        const std::int32_t k = o.k;
        std::int32_t max_t = 0;
        for (auto t : o.tokens) { max_t = std::max(max_t, t); }

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        int dev = 0;
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        std::printf("device %d: %s (sm_%d%d)  -- pin a V100 with CUDA_VISIBLE_DEVICES if this is "
                    "not a V100\n",
                    dev, prop.name, prop.major, prop.minor);

        DeviceBuffer x = bench::make_bf16(static_cast<std::size_t>(k) * max_t);
        DeviceBuffer out_mma(static_cast<std::size_t>(n) * max_t * 2);
        DeviceBuffer out_simt(static_cast<std::size_t>(n) * max_t * 2);

        bench::PackedQuantizedWeight weight =
            bench::make_row_split_weight(QType::Q4G64_F16S, n, k, k);
        randomize_weight(weight, n, k);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Workspace for the multi-split path (n*t*4). Single-split (kDirect) uses none.
        WorkspaceArena ws(static_cast<std::size_t>(n) * max_t * sizeof(float) + (64u << 20));

        std::printf("Q4 volta_mma GEMM  n=%d k=%d\n", n, k);
        for (const std::int32_t t : o.tokens) {
            Tensor xt(x.p, DType::BF16, {k, t});
            Tensor omma(out_mma.p, DType::BF16, {n, t});

            if (o.validate) {
                Tensor osimt(out_simt.p, DType::BF16, {n, t});
                detail::launch_q4_simt_r8_c8(xt, weight.weight, osimt, stream);
                detail::launch_q4_volta_mma(xt, weight.weight, omma, ws, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::vector<std::uint16_t> h_ref(static_cast<std::size_t>(n) * t);
                std::vector<std::uint16_t> h_got(static_cast<std::size_t>(n) * t);
                out_simt.copy_to_host(h_ref.data(), h_ref.size() * 2);
                out_mma.copy_to_host(h_got.data(), h_got.size() * 2);
                const double rel = max_rel_error(h_ref, h_got);
                std::printf("  T=%-5d %-6s max_rel_err=%.4f%%",
                            t, rel < 0.02 ? "PASS" : "FAIL", rel * 100.0);
            } else {
                std::printf("  T=%-5d", t);
            }

            const bench::ColdTiming timing = bench::measure_launch(
                [&](cudaStream_t s) { detail::launch_q4_volta_mma(xt, weight.weight, omma, ws, s); },
                stream, o.warmup, o.repeat);
            const double flops = 2.0 * n * k * t;
            std::printf("   median=%.2f us   %.2f TFLOP/s\n", timing.median_us,
                        flops / (timing.median_us * 1e-6) / 1e12);

            // int8/dp4a route: validate vs the same bf16 SIMT oracle with a LOOSER threshold --
            // int8 activation quant is an approximation (~0.5-1% rel err), not a bit-match, so a
            // correct kernel lands well under 2% while a bug (wrong K-order/scale) blows up.
            {
                DeviceBuffer out_dp4a(static_cast<std::size_t>(n) * max_t * 2);
                Tensor odp(out_dp4a.p, DType::BF16, {n, t});
                std::printf("    dp4a");
                if (o.validate) {
                    Tensor osimt(out_simt.p, DType::BF16, {n, t});
                    detail::launch_q4_simt_r8_c8(xt, weight.weight, osimt, stream);
                    detail::launch_q4_volta_dp4a(xt, weight.weight, odp, ws, stream);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    std::vector<std::uint16_t> h_ref(static_cast<std::size_t>(n) * t);
                    std::vector<std::uint16_t> h_got(static_cast<std::size_t>(n) * t);
                    out_simt.copy_to_host(h_ref.data(), h_ref.size() * 2);
                    out_dp4a.copy_to_host(h_got.data(), h_got.size() * 2);
                    // int8 activation quant: measure error against the output magnitude scale.
                    const double rel = magnitude_rel_error(h_ref, h_got);
                    std::printf(" %-6s mag_rel_err=%.4f%%", rel < 0.01 ? "PASS" : "FAIL",
                                rel * 100.0);
                }
                const bench::ColdTiming td = bench::measure_launch(
                    [&](cudaStream_t s) {
                        detail::launch_q4_volta_dp4a(xt, weight.weight, odp, ws, s);
                    },
                    stream, o.warmup, o.repeat);
                std::printf("   median=%.2f us   %.2f TFLOP/s\n", td.median_us,
                            flops / (td.median_us * 1e-6) / 1e12);
            }
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "q4_volta_mma_bench failed: %s\n", e.what());
        return 1;
    }
}
