// Unsigned integer vectors. Import simd_bytes.simd_int alongside simd_bytes.
namespace simd_int;
import std;

// Eight unsigned 16-bit lanes. Arithmetic wraps modulo 65536; masks use
// zero/65535 lanes. Width does not imply a required CPU instruction set.
//
// ```goose
// let v = simd_int::splat(7);
// assert(v.reduce_max() == 7);
// ```
struct U16x8 { lanes: u16[8] }

extern fn load_low_raw(input: const u32[:]) -> U16x8 = "goosimd_int_load_low";
// Load the low 16 bits of eight u32s, discarding their upper halves.
// Asserts input.len >= 8; reads exactly eight elements.
//
// ```goose
// let input: u32[8] = [65537, 2, 3, 4, 5, 6, 7, 8];
// assert(simd_int::load_low_u16(input).lanes[0] == 1);
// ```
inline fn load_low_u16(input: const u32[:]) -> U16x8 { assert(input.len >= 8); return load_low_raw(input); }

// Fill all eight lanes with value.
//
// ```goose
// assert(simd_int::splat(13).lanes[7] == 13);
// ```
extern fn splat(value: u16) -> U16x8 = "goosimd_int_splat";
// Add corresponding lanes modulo 65536.
//
// ```goose
// assert(simd_int::splat(65535).add(simd_int::splat(2)).lanes[0] == 1);
// ```
extern fn add(a: U16x8, b: U16x8) -> U16x8 = "goosimd_int_add";
// Inclusive prefix sums, wrapping modulo 65536 in each lane.
//
// ```goose
// let v = simd_int::U16x8 { lanes: [2, 1, 4, 3, 8, 5, 6, 7] }.prefix_sum();
// assert(v.lanes[3] == 10 && v.lanes[7] == 36);
// ```
extern fn prefix_sum(v: U16x8) -> U16x8 = "goosimd_int_prefix";
// Broadcast lane 7 to all lanes.
//
// ```goose
// assert(simd_int::splat(3).prefix_sum().broadcast_last().lanes[0] == 24);
// ```
extern fn broadcast_last(v: U16x8) -> U16x8 = "goosimd_int_last";
// Unsigned <= comparison. Matched lanes are 65535; others are zero.
//
// ```goose
// assert(simd_int::splat(40000).le(simd_int::splat(32767)).to_bits() == 0);
// ```
extern fn le(a: U16x8, b: U16x8) -> U16x8 = "goosimd_int_le";
// Extract one bit per mask lane, lane 0 first. Input must contain only
// zero/65535 lanes, such as the result of le(). Upper 24 bits are zero.
//
// ```goose
// assert(simd_int::splat(2).prefix_sum().le(simd_int::splat(6)).to_bits() == 7);
// ```
extern fn to_bits(mask: U16x8) -> u32 = "goosimd_int_bits";
// Select yes on matched mask lanes, no otherwise. Mask lanes must be
// zero/65535, such as the result of le().
//
// ```goose
// let mask = simd_int::splat(1).prefix_sum().le(simd_int::splat(4));
// let v = mask.select(simd_int::splat(19), simd_int::splat(23));
// assert(v.lanes[3] == 19 && v.lanes[4] == 23);
// ```
extern fn select(mask: U16x8, yes: U16x8, no: U16x8) -> U16x8 = "goosimd_int_select";
// Per-lane unsigned minimum.
//
// ```goose
// assert(simd_int::splat(65535).min(simd_int::splat(7)).lanes[0] == 7);
// ```
extern fn min(a: U16x8, b: U16x8) -> U16x8 = "goosimd_int_min";
// Per-lane unsigned maximum.
//
// ```goose
// assert(simd_int::splat(65535).max(simd_int::splat(7)).lanes[7] == 65535);
// ```
extern fn max(a: U16x8, b: U16x8) -> U16x8 = "goosimd_int_max";
// Smallest unsigned lane value.
//
// ```goose
// assert(simd_int::splat(3).prefix_sum().reduce_min() == 3);
// ```
extern fn reduce_min(v: U16x8) -> u16 = "goosimd_int_minimum";
// Largest unsigned lane value.
//
// ```goose
// assert(simd_int::splat(3).prefix_sum().reduce_max() == 24);
// ```
extern fn reduce_max(v: U16x8) -> u16 = "goosimd_int_maximum";

extern fn expand_raw(input: const u32[:], output: u32[:], mask: u16) = "goosimd_int_expand";
// Expand consecutive input values into the set lanes of a 16-bit mask;
// unset lanes become zero. Writes exactly 16 output elements. Returns false
// without writing if output.len < 16 or input.len < popcount(mask).
// Input/output must not overlap; neither requires alignment or extra padding.
//
// ```goose
// let input: u32[3] = [90000, 17, 4294967295];
// var out: u32[16] = [99; 16];
// assert(simd_int::expand_u32(input, out, 0x8005));
// assert(out[0] == 90000 && out[1] == 0 && out[2] == 17 && out[15] == 4294967295);
// ```
fn expand_u32(input: const u32[:], output: u32[:], mask: u16) -> bool {
    var count = 0;
    for i in 16 { if mask & ((1 as u16) << i) != 0 { count += 1; } }
    if input.len < count || output.len < 16 { return false; }
    expand_raw(input, output, mask); true
}
