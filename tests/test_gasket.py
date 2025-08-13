import numpy as np
from PIL import Image

from sulfurvision import pysulfur, util


def test_affine():
    affine = (0.5, 0, 0.5, 0, 0.5, 0)
    coord = (1.5, 0.25)
    print(pysulfur.affine_transform(coord, affine))


def test_gasket():
    t1 = pysulfur.Transform(
        [1, 0, 0, 0, 0, 0, 0], [], (0.5, 0, 0, 0, 0.5, 0), 1.0 / 3, 0
    )
    t2 = pysulfur.Transform(
        [1 / 3, 1 / 3, 0, 0, 0, 1 / 3, 0], [], (0.5, 0, 0.5, 0, 0.5, 0), 1.0 / 3, 1
    )
    t3 = pysulfur.Transform(
        [0, 0.25, 0.125, 0.125, 0.25, 0.125, 0.125],
        [],
        (0.5, 0, 0, 0, 0.5, 0.5),
        1.0 / 3,
        2,
    )
    red = np.array([1, 0, 0, 1])
    green = np.array([0, 1, 0, 1])
    blue = np.array([0, 0, 1, 1])
    wheel = [red, green, blue, red]
    flame = pysulfur.Flame(
        [t1, t2, t3],
        lambda x: util.lerp(wheel[int(x)], wheel[int(x) + 1], x - int(x)),
        (0.5, 0, 0.5, 0, 0.5, 0.5),
    )
    seeds = list(range(1, 12))
    w, h = 400, 400
    array = flame.plot((w, h, 4), seeds, 10000)
    print(array[0, 0], array[0, 0].sum())
    img = Image.new("RGB", (w, h))
    for y in range(h):
        for x in range(w):
            if abs(array[x, y, -1]) > 1e-9:
                img.putpixel(
                    (x, y),
                    tuple((array[x, y, :-1] * 255 / array[x, y, -1]).astype(int)),
                )
    img.save("test_gasket.png")


def main():
    test_gasket()


if __name__ == "__main__":
    main()
