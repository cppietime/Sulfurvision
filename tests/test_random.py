from sulfurvision import pysulfur

from PIL import Image

def test_randfunc(w, h, func, name):
    seed = 123456
    img = Image.new('L', (w, h))
    for y in range(h):
        seed2 = seed + y
        for x in range(w):
            seed2 = func(seed2)
            pix = int(seed2 / 0x100000000 * 256)
            img.putpixel((x, y), pix)
    img.show()
    img.save(f'{name}.png')

def main():
    test_randfunc(100, 100, pysulfur.xorshift32, 'test_xorshift')
    test_randfunc(100, 100, pysulfur.lcg32, 'test_lcg')

if __name__ == '__main__':
    main()