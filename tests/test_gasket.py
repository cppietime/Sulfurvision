from sulfurvision import pysulfur

from PIL import Image

def test_affine():
    affine = (0.5, 0, 0.5, 0, 0.5, 0)
    coord = (1.5, 0.25)
    print(pysulfur.affine_transform(coord, affine))

def test_gasket():
    t1 = pysulfur.Transform([1], [], (0.5, 0, 0, 0, 0.5, 0), 1./3, 1)
    t2 = pysulfur.Transform([1], [], (0.5, 0, 0.5, 0, 0.5, 0), 1./3, 1)
    t3 = pysulfur.Transform([1], [], (0.5, 0, 0, 0, 0.5, 0.5), 1./3, 1)
    flame = pysulfur.Flame([t1, t2, t3])
    seeds = list(range(1, 12))
    w, h = 100, 100
    array = flame.plot((w, h), seeds, 1000)
    img = Image.new('L', (w, h))
    for y in range(h):
        for x in range(w):
            img.putpixel((x, y), int(array[(x, y)] * 255))
    img.save('test_gasket.png')

def main():
    test_gasket()

if __name__ == '__main__':
    main()
