#define MULTIPLIER_LCG32 1664525
#define INCREMENT_LCG32 1013904223
#define MASK32 0xFFFFFFFF
#define EPSILON 1e-9

#define LCG32_UNIFORM(seed, p) seed = lcg32(seed); float p = (float)seed / MASK32;

#define TONEMAP_MODE_LOG 0x1
