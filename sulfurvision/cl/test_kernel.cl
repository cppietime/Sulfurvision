uint xorshift(int seed) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
}

#define MULTIPLIER 1664525
#define INCREMENT 1013904223

uint lcg(uint seed) {
    return seed * MULTIPLIER + INCREMENT;
}

ulong uipow(ulong base, ulong exp) {
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

uint lcg_skip(uint seed, uint skip) {
    /*
    X(n) = ((a^n * X(0) mod M) + ((a^n - 1) mod (a - 1)m / (a - 1))b) mod M
    */
    ulong a_to_n = uipow(MULTIPLIER, skip);
    ulong a_minus_one = skip - 1;
    ulong new_seed = (a_to_n * seed) & 0xFFFFFFFF;
    new_seed += INCREMENT * (((a_to_n - 1) % ((a_minus_one) << 32)) / a_minus_one);
    return new_seed & 0xFFFFFFFF;
}

__kernel void test_kernel(__global float* img, const int width, const int height) {
    uint id = get_global_id(0);
    uint x = id % width;
    uint y = id / width;
    uint seed = 12345;
    seed = lcg_skip(seed, id << 8);
    uint prng = lcg(seed);
    img[id] = (float)prng / (float)0xFFFFFFFF;
}
