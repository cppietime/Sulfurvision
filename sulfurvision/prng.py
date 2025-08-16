def xorshift32(seed: int) -> int:
    seed ^= (seed << 13) & 0xFFFFFFFF
    seed ^= (seed >> 17) & 0xFFFFFFFF
    seed ^= (seed << 5) & 0xFFFFFFFF
    return seed

_mul = 1664525
_inc = 1013904223
def lcg32(seed: int) -> int:
    return (seed * _mul + _inc) & 0xFFFFFFFF

def lcg32_skip(seed: int, skip: int) -> int:
    a_to_n = pow(_mul, skip, 1 << 64)
    val = (
        ((a_to_n * seed) & 0xFFFFFFFF) +
        (((a_to_n - 1) % ((_mul - 1) << 32)) // (_mul - 1)) * _inc
    ) & 0xFFFFFFFF
    return val

_randfunc = xorshift32


def rand_u32(seed: int) -> int:
    return _randfunc(seed)


def rand_uniform(seed: int, scale: float = 1.0) -> tuple[int, float]:
    new_seed = rand_u32(seed)
    return new_seed, new_seed / 0x100000000 * scale
