def xorshift32(seed: int) -> int:
    seed ^= (seed << 13) & 0xFFFFFFFF
    seed ^= (seed >> 17) & 0xFFFFFFFF
    seed ^= (seed << 5) & 0xFFFFFFFF
    return seed


def lcg32(seed: int) -> int:
    mul = 1664525
    inc = 1013904223
    return (seed * mul + inc) & 0xFFFFFFFF


_randfunc = xorshift32


def rand_u32(seed: int) -> int:
    return _randfunc(seed)


def rand_uniform(seed: int, scale: float = 1.0) -> tuple[int, float]:
    new_seed = rand_u32(seed)
    return new_seed, new_seed / 0x100000000 * scale
