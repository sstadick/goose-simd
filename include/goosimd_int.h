#ifndef GOOSIMD_INT_H
#define GOOSIMD_INT_H
#include <stdint.h>
#include <string.h>
#include "goosimd_expand.h"

static inline void goosimd_int_expand(sl_u32 input, sl_u32 output, uint16_t mask) {
    goosimd_expand_u32(input.data, output.data, mask);
}

#if defined(__clang__) && !defined(__TINYC__) && !defined(GOOSIMD_SCALAR)
#define GSI_INLINE static inline __attribute__((always_inline))
typedef uint16_t gsi_u16x8 __attribute__((vector_size(16)));
#define GSI_VECTOR 1
#else
#define GSI_INLINE static inline
typedef struct { uint16_t lane[8]; } gsi_u16x8;
#define GSI_VECTOR 0
#endif

GSI_INLINE gsi_u16x8 gsi_splat(uint16_t value) {
    uint16_t lanes[8]; for (unsigned i = 0; i < 8; ++i) lanes[i] = value;
    gsi_u16x8 result; memcpy(&result, lanes, 16); return result;
}
GSI_INLINE gsi_u16x8 gsi_load_low(const void *input) {
    uint32_t words[8]; memcpy(words, input, sizeof words);
    uint16_t lanes[8]; for (unsigned i = 0; i < 8; ++i) lanes[i] = (uint16_t)words[i];
    gsi_u16x8 result; memcpy(&result, lanes, 16); return result;
}
#if GSI_VECTOR
#define GSI_BINARY(name, op) GSI_INLINE gsi_u16x8 gsi_##name(gsi_u16x8 a, gsi_u16x8 b) { return (gsi_u16x8)(a op b); }
#else
#define GSI_BINARY(name, op) GSI_INLINE gsi_u16x8 gsi_##name(gsi_u16x8 a, gsi_u16x8 b) { \
    gsi_u16x8 r; for (unsigned i = 0; i < 8; ++i) r.lane[i] = (uint16_t)(a.lane[i] op b.lane[i]); return r; }
#endif
GSI_BINARY(add, +)
#undef GSI_BINARY

GSI_INLINE gsi_u16x8 gsi_le(gsi_u16x8 a, gsi_u16x8 b) {
#if GSI_VECTOR
    return (gsi_u16x8)(a <= b);
#else
    gsi_u16x8 r; for (unsigned i = 0; i < 8; ++i) r.lane[i] = a.lane[i] <= b.lane[i] ? 65535 : 0; return r;
#endif
}
GSI_INLINE gsi_u16x8 gsi_select(gsi_u16x8 mask, gsi_u16x8 yes, gsi_u16x8 no) {
#if GSI_VECTOR
    return (mask & yes) | (~mask & no);
#else
    gsi_u16x8 r; for (unsigned i = 0; i < 8; ++i) r.lane[i] = mask.lane[i] ? yes.lane[i] : no.lane[i]; return r;
#endif
}
GSI_INLINE gsi_u16x8 gsi_min(gsi_u16x8 a, gsi_u16x8 b) { return gsi_select(gsi_le(a, b), a, b); }
GSI_INLINE gsi_u16x8 gsi_max(gsi_u16x8 a, gsi_u16x8 b) { return gsi_select(gsi_le(a, b), b, a); }
GSI_INLINE gsi_u16x8 gsi_prefix(gsi_u16x8 x) {
#if GSI_VECTOR
    gsi_u16x8 zero = {0};
    x += __builtin_shufflevector(zero, x, 0,8,9,10,11,12,13,14);
    x += __builtin_shufflevector(zero, x, 0,0,8,9,10,11,12,13);
    return x + __builtin_shufflevector(zero, x, 0,0,0,0,8,9,10,11);
#else
    for (unsigned i = 1; i < 8; ++i) x.lane[i] = (uint16_t)(x.lane[i] + x.lane[i-1]); return x;
#endif
}
GSI_INLINE gsi_u16x8 gsi_last(gsi_u16x8 x) {
#if GSI_VECTOR
    return __builtin_shufflevector(x, x, 7,7,7,7,7,7,7,7);
#else
    return gsi_splat(x.lane[7]);
#endif
}
GSI_INLINE uint32_t gsi_bits(gsi_u16x8 mask) {
#if GSI_VECTOR
    typedef _Bool bool8 __attribute__((ext_vector_type(8)));
    bool8 bits = __builtin_convertvector(mask, bool8);
    uint8_t result; memcpy(&result, &bits, 1); return result;
#else
    uint32_t result = 0; for (unsigned i = 0; i < 8; ++i) result |= (uint32_t)(mask.lane[i] != 0) << i; return result;
#endif
}
GSI_INLINE uint16_t gsi_minimum(gsi_u16x8 x) {
#if GSI_VECTOR
    return __builtin_reduce_min(x);
#else
    uint16_t r = 65535; for (unsigned i = 0; i < 8; ++i) if (x.lane[i] < r) r = x.lane[i]; return r;
#endif
}
GSI_INLINE uint16_t gsi_maximum(gsi_u16x8 x) {
#if GSI_VECTOR
    return __builtin_reduce_max(x);
#else
    uint16_t r = 0; for (unsigned i = 0; i < 8; ++i) if (x.lane[i] > r) r = x.lane[i]; return r;
#endif
}

// As with goosimd.h, these adapters follow Goose's generated packed types.
typedef simd_int_U16x8_g8 simd_int_U16x8;
GSI_INLINE simd_int_U16x8 goosimd_int_splat(uint16_t x) {
    gsi_u16x8 v = gsi_splat(x); simd_int_U16x8 r; memcpy(&r, &v, 16); return r;
}
GSI_INLINE simd_int_U16x8 goosimd_int_load_low(sl_u32 input) {
    gsi_u16x8 v = gsi_load_low(input.data); simd_int_U16x8 r; memcpy(&r, &v, 16); return r;
}
#define GSI_ADAPT_BINARY(name) \
GSI_INLINE simd_int_U16x8 goosimd_int_##name(simd_int_U16x8 a, simd_int_U16x8 b) { \
    gsi_u16x8 x, y; memcpy(&x, &a, 16); memcpy(&y, &b, 16); x = gsi_##name(x, y); memcpy(&a, &x, 16); return a; }
GSI_ADAPT_BINARY(add)
GSI_ADAPT_BINARY(le)
GSI_ADAPT_BINARY(min)
GSI_ADAPT_BINARY(max)
#undef GSI_ADAPT_BINARY
#define GSI_ADAPT_UNARY(name, type) \
GSI_INLINE type goosimd_int_##name(simd_int_U16x8 a) { \
    gsi_u16x8 x; memcpy(&x, &a, 16); return gsi_##name(x); }
GSI_ADAPT_UNARY(bits, uint32_t)
GSI_ADAPT_UNARY(minimum, uint16_t)
GSI_ADAPT_UNARY(maximum, uint16_t)
#undef GSI_ADAPT_UNARY
GSI_INLINE simd_int_U16x8 goosimd_int_prefix(simd_int_U16x8 a) {
    gsi_u16x8 x; memcpy(&x, &a, 16); x = gsi_prefix(x); memcpy(&a, &x, 16); return a;
}
GSI_INLINE simd_int_U16x8 goosimd_int_last(simd_int_U16x8 a) {
    gsi_u16x8 x; memcpy(&x, &a, 16); x = gsi_last(x); memcpy(&a, &x, 16); return a;
}
GSI_INLINE simd_int_U16x8 goosimd_int_select(simd_int_U16x8 m, simd_int_U16x8 a, simd_int_U16x8 b) {
    gsi_u16x8 mask, x, y; memcpy(&mask, &m, 16); memcpy(&x, &a, 16); memcpy(&y, &b, 16);
    x = gsi_select(mask, x, y); memcpy(&a, &x, 16); return a;
}
#undef GSI_INLINE
#undef GSI_VECTOR
#endif
