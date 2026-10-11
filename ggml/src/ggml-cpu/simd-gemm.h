#pragma once

// Computes C[M x N] += A[M x K] * B[K x N]

#include "simd-mappings.h"

// TODO: add support for sizeless vector types
#if defined(GGML_SIMD) && !defined(__ARM_FEATURE_SVE) && !defined(__riscv_v_intrinsic)

// TODO: untested on avx512
// These are in units of GGML_F32_EPR
#if defined(__AVX512F__) || defined (__ARM_NEON__)
    static constexpr int GEMM_RM = 4;
    static constexpr int GEMM_RN = 4; // 16+4+1 = 25/32
#elif defined(__AVX2__) || defined(__AVX__)
    static constexpr int GEMM_RM = 6;
    static constexpr int GEMM_RN = 2; // 12+2+1 = 15/16
#else
    static constexpr int GEMM_RM = 2;
    static constexpr int GEMM_RN = 2;
#endif

template <int RM, int RN>
static inline void simd_gemm_ukernel(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N)
{
    static constexpr int KN = GGML_F32_EPR;

    GGML_F32_VEC acc[RM][RN];
    for (int64_t i = 0; i < RM; i++) {
        for (int r = 0; r < RN; r++) {
            acc[i][r] = GGML_F32_VEC_LOAD(C + i * N + r * KN);
        }
    }

    for (int64_t kk = 0; kk < K; kk++) {
        GGML_F32_VEC Bv[RN];
        for (int r = 0; r < RN; r++) {
            Bv[r] = GGML_F32_VEC_LOAD(B + kk * N + r * KN);
        }
        for (int64_t i = 0; i < RM; i++) {
            GGML_F32_VEC p = GGML_F32_VEC_SET1(A[i * K + kk]);
            for (int r = 0; r < RN; r++) {
                acc[i][r] = GGML_F32_VEC_FMA(acc[i][r], Bv[r], p);
            }
        }
    }

    for (int64_t i = 0; i < RM; i++) {
        for (int r = 0; r < RN; r++) {
            GGML_F32_VEC_STORE(C + i * N + r * KN, acc[i][r]);
        }
    }
}

// C[M x N] += A[M x K] * B[K x N]
static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int M, int K, int N)
{
    static constexpr int KN = GGML_F32_EPR;

    int64_t ii = 0;
    for (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        for (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<GEMM_RM, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        for (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<GEMM_RM, 1>(C + jj, A, B + jj, K, N);
        }
        for (; jj < N; jj++) {
            for (int64_t i = 0; i < GEMM_RM; i++) {
                float a = C[i * N + jj];
                for (int64_t kk = 0; kk < K; kk++) {
                    a += A[i * K + kk] * B[kk * N + jj];
                }
                C[i * N + jj] = a;
            }
        }

        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    // Tail rows: one at a time
    for (; ii < M; ii++) {
        int64_t jj = 0;
        for (; jj + GEMM_RN * KN <= N; jj += GEMM_RN * KN) {
            simd_gemm_ukernel<1, GEMM_RN>(C + jj, A, B + jj, K, N);
        }
        for (; jj + KN <= N; jj += KN) {
            simd_gemm_ukernel<1, 1>(C + jj, A, B + jj, K, N);
        }
        for (; jj < N; jj++) {
            float a = C[jj];
            for (int64_t kk = 0; kk < K; kk++) {
                a += A[kk] * B[kk * N + jj];
            }
            C[jj] = a;
        }

        A += K;
        C += N;
    }
}
#elif defined(GGML_SIMD) && defined(__riscv_v_intrinsic)
// RM accumulators + 1 B vector = RM + 1 <= 8  =>  RM <= 7
// Microkernel: C[RM x vl] += A[RM x K] * B[K x N]
template <int RM>
static inline void rvv_simd_gemm_ukernel(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N, size_t vl)
{
    static_assert(RM >= 1 && RM <= 7, "RM must be 1..7 for LMUL=4");

    vfloat32m4_t acc_0 = __riscv_vle32_v_f32m4(C + 0 * N, vl);
    vfloat32m4_t acc_1, acc_2, acc_3, acc_4, acc_5, acc_6;
    if constexpr (RM > 1) acc_1 = __riscv_vle32_v_f32m4(C + 1 * N, vl);
    if constexpr (RM > 2) acc_2 = __riscv_vle32_v_f32m4(C + 2 * N, vl);
    if constexpr (RM > 3) acc_3 = __riscv_vle32_v_f32m4(C + 3 * N, vl);
    if constexpr (RM > 4) acc_4 = __riscv_vle32_v_f32m4(C + 4 * N, vl);
    if constexpr (RM > 5) acc_5 = __riscv_vle32_v_f32m4(C + 5 * N, vl);
    if constexpr (RM > 6) acc_6 = __riscv_vle32_v_f32m4(C + 6 * N, vl);

    for (int kk = 0; kk < K; kk++) {
        vfloat32m4_t b_0 = __riscv_vle32_v_f32m4(B + kk * N, vl);

                              acc_0 = __riscv_vfmacc_vf_f32m4(acc_0, A[0 * K + kk], b_0, vl);
        if constexpr (RM > 1) acc_1 = __riscv_vfmacc_vf_f32m4(acc_1, A[1 * K + kk], b_0, vl);
        if constexpr (RM > 2) acc_2 = __riscv_vfmacc_vf_f32m4(acc_2, A[2 * K + kk], b_0, vl);
        if constexpr (RM > 3) acc_3 = __riscv_vfmacc_vf_f32m4(acc_3, A[3 * K + kk], b_0, vl);
        if constexpr (RM > 4) acc_4 = __riscv_vfmacc_vf_f32m4(acc_4, A[4 * K + kk], b_0, vl);
        if constexpr (RM > 5) acc_5 = __riscv_vfmacc_vf_f32m4(acc_5, A[5 * K + kk], b_0, vl);
        if constexpr (RM > 6) acc_6 = __riscv_vfmacc_vf_f32m4(acc_6, A[6 * K + kk], b_0, vl);
    }

                          __riscv_vse32_v_f32m4(C + 0 * N, acc_0, vl);
    if constexpr (RM > 1) __riscv_vse32_v_f32m4(C + 1 * N, acc_1, vl);
    if constexpr (RM > 2) __riscv_vse32_v_f32m4(C + 2 * N, acc_2, vl);
    if constexpr (RM > 3) __riscv_vse32_v_f32m4(C + 3 * N, acc_3, vl);
    if constexpr (RM > 4) __riscv_vse32_v_f32m4(C + 4 * N, acc_4, vl);
    if constexpr (RM > 5) __riscv_vse32_v_f32m4(C + 5 * N, acc_5, vl);
    if constexpr (RM > 6) __riscv_vse32_v_f32m4(C + 6 * N, acc_6, vl);
}

template <int RM>
static inline void rvv_simd_gemm_dispatch_tail(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int K, int N, int KN, int remaining_rows)
{
    if constexpr (RM > 0) {
        if (remaining_rows == RM) {
            int64_t jj = 0;
            for (; jj + KN <= N; jj += KN) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, KN);
            }
            if (jj < N) {
                rvv_simd_gemm_ukernel<RM>(C + jj, A, B + jj, K, N, N - jj);
            }
        } else {
            rvv_simd_gemm_dispatch_tail<RM - 1>(C, A, B, K, N, KN, remaining_rows);
        }
    }
}

static constexpr int GEMM_RM = 7;

// C[M x N] += A[M x K] * B[K x N]
static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int M, int K, int N)
{
    const int KN = (int)__riscv_vlenb();
    int64_t ii = 0;
    for (; ii + GEMM_RM <= M; ii += GEMM_RM) {
        int64_t jj = 0;
        for (; jj + KN <= N; jj += KN) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, KN);
        }
        if (jj < N) {
            rvv_simd_gemm_ukernel<GEMM_RM>(C + jj, A, B + jj, K, N, N - jj);
        }
        A += GEMM_RM * K;
        C += GEMM_RM * N;
    }

    int remaining_rows = M - ii;
    rvv_simd_gemm_dispatch_tail<GEMM_RM - 1>(C, A, B, K, N, KN, remaining_rows);
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#else // scalar path

static void simd_gemm(
    float       * GGML_RESTRICT C,
    const float * GGML_RESTRICT A,
    const float * GGML_RESTRICT B,
    int M, int K, int N)
{
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = C[i * N + j];
            for (int64_t kk = 0; kk < K; kk++) {
                sum += A[i * K + kk] * B[kk * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

#endif // GGML_SIMD

// ---------------------------------------------------------------------------
// F16 GEMM variant (NEON fullfp16): C[M x N] += A[M x K] * B[K x N], all fp16.
// For the flash-attention tiled path on F16 KV: A77-class cores run 8-wide
// fp16 FMA — and the fp16 numeric channel is what upstream's one_chunk path
// already uses for KQ scores (GGML_F16_VEC vfmaq_f16 accumulation).
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define GGML_SIMD_GEMM_F16 1

// 4x4 accumulators x float16x8 + 4 B + 4 A-broadcast = 24 of 32 NEON regs.
static constexpr int GEMM16_RM = 4;
static constexpr int GEMM16_RN = 4;
static constexpr int GEMM16_KN = 8;

// Row pitches are explicit: the flash-attn tiled path also runs clipped
// GEMMs, where the C/B/A row stride differs from N/K (the K tile / KQh are
// packed with a KV_TILE_SZ pitch while N is the clipped column count).
template <int RM, int RN>
static inline void simd_gemm16_ukernel(
    ggml_fp16_t       * GGML_RESTRICT C,
    const ggml_fp16_t * GGML_RESTRICT A,
    const ggml_fp16_t * GGML_RESTRICT B,
    int K, int lda, int ldb, int ldc)
{
    float16x8_t acc[RM][RN];
    for (int i = 0; i < RM; i++) {
        for (int r = 0; r < RN; r++) {
            acc[i][r] = vld1q_f16((const __fp16 *)(C + i * ldc + r * GEMM16_KN));
        }
    }

    for (int kk = 0; kk < K; kk++) {
        float16x8_t Bv[RN];
        for (int r = 0; r < RN; r++) {
            Bv[r] = vld1q_f16((const __fp16 *)(B + kk * ldb + r * GEMM16_KN));
        }
        float16x8_t av[RM];
        for (int i = 0; i < RM; i++) {
            av[i] = vdupq_n_f16(*(const __fp16 *)(A + i * lda + kk));
        }
        for (int i = 0; i < RM; i++) {
            for (int r = 0; r < RN; r++) {
                acc[i][r] = vfmaq_f16(acc[i][r], Bv[r], av[i]);
            }
        }
    }

    for (int i = 0; i < RM; i++) {
        for (int r = 0; r < RN; r++) {
            vst1q_f16((__fp16 *)(C + i * ldc + r * GEMM16_KN), acc[i][r]);
        }
    }
}

// C[M x N] += A[M x K] * B[K x N]   (all fp16)
// lda/ldb/ldc are the A/B/C row pitches in elements; <= 0 selects the dense
// pitch (K/N/N), which is what every dense caller wants.
static void simd_gemm_f16(
    ggml_fp16_t       * GGML_RESTRICT C,
    const ggml_fp16_t * GGML_RESTRICT A,
    const ggml_fp16_t * GGML_RESTRICT B,
    int M, int K, int N, int lda = -1, int ldb = -1, int ldc = -1)
{
    if (lda <= 0) { lda = K; }
    if (ldb <= 0) { ldb = N; }
    if (ldc <= 0) { ldc = N; }

    int64_t ii = 0;
    for (; ii + GEMM16_RM <= M; ii += GEMM16_RM) {
        int64_t jj = 0;
        for (; jj + GEMM16_RN * GEMM16_KN <= N; jj += GEMM16_RN * GEMM16_KN) {
            simd_gemm16_ukernel<GEMM16_RM, GEMM16_RN>(C + jj, A, B + jj, K, lda, ldb, ldc);
        }
        for (; jj + GEMM16_KN <= N; jj += GEMM16_KN) {
            simd_gemm16_ukernel<GEMM16_RM, 1>(C + jj, A, B + jj, K, lda, ldb, ldc);
        }
        for (; jj < N; jj++) {
            for (int i = 0; i < GEMM16_RM; i++) {
                float a = GGML_FP16_TO_FP32(C[i * ldc + jj]);
                for (int kk = 0; kk < K; kk++) {
                    a += GGML_FP16_TO_FP32(A[i * lda + kk]) * GGML_FP16_TO_FP32(B[kk * ldb + jj]);
                }
                C[i * ldc + jj] = GGML_FP32_TO_FP16(a);
            }
        }

        A += GEMM16_RM * lda;
        C += GEMM16_RM * ldc;
    }

    // Tail rows: one at a time
    for (; ii < M; ii++) {
        int64_t jj = 0;
        for (; jj + GEMM16_RN * GEMM16_KN <= N; jj += GEMM16_RN * GEMM16_KN) {
            simd_gemm16_ukernel<1, GEMM16_RN>(C + jj, A, B + jj, K, lda, ldb, ldc);
        }
        for (; jj + GEMM16_KN <= N; jj += GEMM16_KN) {
            simd_gemm16_ukernel<1, 1>(C + jj, A, B + jj, K, lda, ldb, ldc);
        }
        for (; jj < N; jj++) {
            float a = GGML_FP16_TO_FP32(C[jj]);
            for (int kk = 0; kk < K; kk++) {
                a += GGML_FP16_TO_FP32(A[kk]) * GGML_FP16_TO_FP32(B[kk * ldb + jj]);
            }
            C[jj] = GGML_FP32_TO_FP16(a);
        }

        A += lda;
        C += ldc;
    }
}
#else
// Scalar placeholder so simd_gemm_f16 always resolves at compile time.
// It is dead code unless a target defines GGML_SIMD_GEMM_F16 (the tiled
// flash-attn path gates on that macro, use_gemm_f16 stays false otherwise).
static void simd_gemm_f16(
    ggml_fp16_t       * GGML_RESTRICT C,
    const ggml_fp16_t * GGML_RESTRICT A,
    const ggml_fp16_t * GGML_RESTRICT B,
    int M, int K, int N, int lda = -1, int ldb = -1, int ldc = -1)
{
    if (lda <= 0) { lda = K; }
    if (ldb <= 0) { ldb = N; }
    if (ldc <= 0) { ldc = N; }

    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = GGML_FP16_TO_FP32(C[i * ldc + j]);
            for (int64_t kk = 0; kk < K; kk++) {
                sum += GGML_FP16_TO_FP32(A[i * lda + kk]) * GGML_FP16_TO_FP32(B[kk * ldb + j]);
            }
            C[i * ldc + j] = GGML_FP32_TO_FP16(sum);
        }
    }
}
#endif // __ARM_NEON && __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
