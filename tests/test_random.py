from PIL import Image
from sulfurvision import prng

def seeded_xor(seed):
    a = 0x2491023
    b = 0x94201309
    seed += a
    seed ^= seed << 11
    return (b ^ (b >> 19) ^ seed ^ (seed >> 8)) & 0xFFFFFFFF

def java(seed):
    r = 0x2301340913408135
    seed += r
    seed = (seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
    return seed >> 16

def test_randfunc(w, h, func, name):
    seed = 123456
    img = Image.new("L", (w, h))
    for y in range(h):
        seed2 = seed + y
        for x in range(w):
            value = func(seed2 + x * h)
            pix = int(value / 0x100000000 * 256)
            img.putpixel((x, y), pix)
    img.save(f"{name}.png")


def main():
    test_randfunc(100, 100, prng.xorshift32, "test_xorshift")
    test_randfunc(100, 100, prng.lcg32, "test_lcg")
    test_randfunc(100, 100, seeded_xor, "test_sxor")
    test_randfunc(100, 100, java, "test_java")


if __name__ == "__main__":
    main()
