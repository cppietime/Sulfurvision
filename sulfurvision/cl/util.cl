uint lcg32(uint seed) {
    return seed * MULTIPLIER_LCG32 + INCREMENT_LCG32;
}

ulong uipow64(ulong base, ulong exp) {
    ulong result = 1;
    while (exp) {
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
        base *= base;
    }
    return result;
}

uint lcg32_skip(uint seed, uint skip) {
    /*
    X(n) = ((a^n * X(0) mod M) + ((a^n - 1) mod (a - 1)m / (a - 1))b) mod M
    */
    ulong a_to_n = uipow64(MULTIPLIER_LCG32, skip);
    ulong a_minus_one = skip - 1;
    ulong new_seed = (a_to_n * seed) & MASK32;
    new_seed += INCREMENT_LCG32 * (((a_to_n - 1) % ((a_minus_one) << 32)) / a_minus_one);
    return new_seed & MASK32;
}

float2 affine_transform(__constant float* affine, float2 xy) {
    return (float2)(xy.x * affine[0] + xy.y * affine[1] + affine[2], xy.x * affine[3] + xy.y * affine[4] + affine[5]);
}

uchar4 sample_palette(__constant float4* palette, const float color, const uint n_colors) {
    uint idx0 = floor(color);
    float4 color0 = palette[idx0];
    float frac = color - idx0;
    if (frac > EPSILON) {
        float4 color1 = palette[idx0 + 1];
        color0 += (color1 - color0) * frac;
    }
    return (uchar4)(color0.x, color0.y, color0.z, color0.w);
}
