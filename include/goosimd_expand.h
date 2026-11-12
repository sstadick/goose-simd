#ifndef GOOSIMD_EXPAND_H
#define GOOSIMD_EXPAND_H
#include <stdint.h>
#include <string.h>

// Zero-fill expansion: consume consecutive input lanes at set mask bits.
// Target-specific helpers are for callers already inside a dispatched kernel.
#if defined(__clang__) && !defined(__TINYC__) && !defined(GOOSIMD_SCALAR) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define GSE_X86 1
__attribute__((target("avx512f"), always_inline))
static inline __m512i goosimd_expand_u32x16(__m512i values, unsigned mask) {
    return _mm512_maskz_expand_epi32((__mmask16)mask, values);
}
__attribute__((target("avx512f")))
static void gse_expand16(const uint32_t *input, uint32_t *output, unsigned mask) {
    _mm512_storeu_si512(output, goosimd_expand_u32x16(_mm512_loadu_si512(input), mask));
}
#elif defined(__aarch64__) && !defined(__TINYC__) && !defined(GOOSIMD_SCALAR)
#include <arm_neon.h>
#define GSE_NEON 1
#define GSE_BYTE(m,l,b) ((m) & (1u << (l)) ? 4 * __builtin_popcount((m) & ((1u << (l))-1)) + (b) : 128)
#define GSE_LANE(m,l) GSE_BYTE(m,l,0), GSE_BYTE(m,l,1), GSE_BYTE(m,l,2), GSE_BYTE(m,l,3)
#define GSE_ROW(m) { GSE_LANE(m,0), GSE_LANE(m,1), GSE_LANE(m,2), GSE_LANE(m,3) }
static const uint8_t gse_shuffle[16][16] = {
    GSE_ROW(0), GSE_ROW(1), GSE_ROW(2), GSE_ROW(3), GSE_ROW(4), GSE_ROW(5), GSE_ROW(6), GSE_ROW(7),
    GSE_ROW(8), GSE_ROW(9), GSE_ROW(10), GSE_ROW(11), GSE_ROW(12), GSE_ROW(13), GSE_ROW(14), GSE_ROW(15)
};
#undef GSE_ROW
#undef GSE_LANE
#undef GSE_BYTE
static inline uint32x4_t goosimd_expand_u32x4(uint32x4_t values, unsigned mask) {
    return vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(values), vld1q_u8(gse_shuffle[mask & 15])));
}
#endif

// input has popcount(mask) elements, output has 16. No alignment or readable
// slack required; input and output must not overlap.
static inline void goosimd_expand_u32(const void *input, void *output, unsigned mask) {
    mask &= 65535;
#if defined(GSE_X86) || defined(GSE_NEON)
    uint32_t padded[16] = {0};
    unsigned count = (unsigned)__builtin_popcount(mask);
    if (count) memcpy(padded, input, count * sizeof(uint32_t));
#ifdef GSE_X86
    if (__builtin_cpu_supports("avx512f")) { gse_expand16(padded, output, mask); return; }
#else
    unsigned used = 0;
    for (unsigned i = 0; i < 16; i += 4) {
        unsigned bits = (mask >> i) & 15;
        // At most twelve inputs precede the last group, even for a full mask.
        vst1q_u32((uint32_t *)((uint8_t *)output + 4*i), goosimd_expand_u32x4(vld1q_u32(padded + used), bits));
        used += (unsigned)__builtin_popcount(bits);
    }
    return;
#endif
#endif
    unsigned cursor = 0;
    for (unsigned i = 0; i < 16; ++i) {
        uint32_t value = 0;
        if (mask & (1u << i)) { memcpy(&value, (const uint8_t *)input + 4*cursor, 4); ++cursor; }
        memcpy((uint8_t *)output + 4*i, &value, 4);
    }
}
#undef GSE_X86
#undef GSE_NEON
#endif
