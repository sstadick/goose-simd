#ifndef GOOSIMD_NIBBLES_H
#define GOOSIMD_NIBBLES_H
#include <stdint.h>
#include <stddef.h>

// Shared native kernel, independent of Goose's generated type names. Input
// and output must not overlap. Callers check ceil(length/2) input bytes and
// provide a 16-byte lookup table. No padding reads or writes.
#if defined(__clang__) && !defined(__TINYC__) && !defined(GOOSIMD_SCALAR) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define GSN_X86 1
#define GSN_TARGET(features) __attribute__((target(features)))
GSN_TARGET("ssse3")
static inline void gsn_block16(const uint8_t *input, uint8_t *out, const uint8_t *lut, int low_first) {
    __m128i x = _mm_loadu_si128((const __m128i *)input);
    __m128i table = _mm_loadu_si128((const __m128i *)lut), mask = _mm_set1_epi8(15);
    __m128i lo = _mm_shuffle_epi8(table, _mm_and_si128(x, mask));
    __m128i hi = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(x, 4), mask));
    __m128i a = low_first ? lo : hi, b = low_first ? hi : lo;
    _mm_storeu_si128((__m128i *)out, _mm_unpacklo_epi8(a, b));
    _mm_storeu_si128((__m128i *)(out + 16), _mm_unpackhi_epi8(a, b));
}
GSN_TARGET("avx2")
static inline void gsn_block32(const uint8_t *input, uint8_t *out, const uint8_t *lut, int low_first) {
    __m256i x = _mm256_loadu_si256((const __m256i *)input);
    __m256i table = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)lut)), mask = _mm256_set1_epi8(15);
    __m256i lo = _mm256_shuffle_epi8(table, _mm256_and_si256(x, mask));
    __m256i hi = _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(x, 4), mask));
    __m256i a = low_first ? lo : hi, b = low_first ? hi : lo;
    lo = _mm256_unpacklo_epi8(a, b); hi = _mm256_unpackhi_epi8(a, b);
    _mm256_storeu_si256((__m256i *)out, _mm256_permute2x128_si256(lo, hi, 0x20));
    _mm256_storeu_si256((__m256i *)(out + 32), _mm256_permute2x128_si256(lo, hi, 0x31));
}
GSN_TARGET("avx512f,avx512bw,avx512dq,avx512vl")
static inline void gsn_block64(const uint8_t *input, uint8_t *out, const uint8_t *lut, int low_first) {
    __m512i x = _mm512_loadu_si512(input);
    __m512i table = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)lut)), mask = _mm512_set1_epi8(15);
    __m512i lo = _mm512_shuffle_epi8(table, _mm512_and_si512(x, mask));
    __m512i hi = _mm512_shuffle_epi8(table, _mm512_and_si512(_mm512_srli_epi16(x, 4), mask));
    __m512i a = low_first ? lo : hi, b = low_first ? hi : lo;
    lo = _mm512_unpacklo_epi8(a, b); hi = _mm512_unpackhi_epi8(a, b);
    _mm512_storeu_si512(out, _mm512_permutex2var_epi64(lo, _mm512_setr_epi64(0,1,8,9,2,3,10,11), hi));
    _mm512_storeu_si512(out + 64, _mm512_permutex2var_epi64(lo, _mm512_setr_epi64(4,5,12,13,6,7,14,15), hi));
}
#define GSN_LOOP(N, FEATURES) \
GSN_TARGET(FEATURES) static size_t gsn_loop##N(const uint8_t *in, uint8_t *out, const uint8_t *lut, size_t pairs, int low) { \
    size_t i = 0; for (; pairs - i >= N; i += N) gsn_block##N(in + i, out + 2*i, lut, low); \
    if (i && i < pairs) { gsn_block##N(in + pairs - N, out + 2*(pairs - N), lut, low); i = pairs; } return i; }
GSN_LOOP(16, "ssse3")
GSN_LOOP(32, "avx2")
GSN_LOOP(64, "avx512f,avx512bw,avx512dq,avx512vl")
#undef GSN_LOOP
#undef GSN_TARGET
#elif defined(__aarch64__) && !defined(__TINYC__) && !defined(GOOSIMD_SCALAR)
#include <arm_neon.h>
#define GSN_NEON 1
static inline void gsn_block16(const uint8_t *input, uint8_t *out, const uint8_t *lut, int low_first) {
    uint8x16_t x = vld1q_u8(input), table = vld1q_u8(lut);
    uint8x16_t lo = vqtbl1q_u8(table, vandq_u8(x, vdupq_n_u8(15)));
    uint8x16_t hi = vqtbl1q_u8(table, vshrq_n_u8(x, 4));
    uint8x16_t a = low_first ? lo : hi, b = low_first ? hi : lo;
    vst1q_u8(out, vzip1q_u8(a, b)); vst1q_u8(out + 16, vzip2q_u8(a, b));
}
static size_t gsn_loop16(const uint8_t *in, uint8_t *out, const uint8_t *lut, size_t pairs, int low) {
    size_t i = 0; for (; pairs - i >= 16; i += 16) gsn_block16(in + i, out + 2*i, lut, low);
    if (i && i < pairs) { gsn_block16(in + pairs - 16, out + 2*(pairs - 16), lut, low); i = pairs; } return i;
}
#endif

static inline void goosimd_decode_nibbles(const uint8_t *input, uint8_t *out, size_t length, const uint8_t lut[16], int low_first) {
    size_t pairs = length / 2, i = 0;
#ifdef GSN_X86
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl"))
        i = gsn_loop64(input, out, lut, pairs, low_first);
    else if (__builtin_cpu_supports("avx2")) i = gsn_loop32(input, out, lut, pairs, low_first);
    else if (__builtin_cpu_supports("ssse3")) i = gsn_loop16(input, out, lut, pairs, low_first);
#elif defined(GSN_NEON)
    i = gsn_loop16(input, out, lut, pairs, low_first);
#endif
    for (; i < pairs; ++i) {
        uint8_t x = input[i];
        out[2*i] = lut[low_first ? x & 15 : x >> 4];
        out[2*i+1] = lut[low_first ? x >> 4 : x & 15];
    }
    if (length & 1) out[length - 1] = lut[low_first ? input[pairs] & 15 : input[pairs] >> 4];
}
#undef GSN_X86
#undef GSN_NEON
#endif
