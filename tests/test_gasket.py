import numpy as np
from PIL import Image

from sulfurvision import pysulfur, util, variations


def test_affine():
    affine = np.array((0.5, 0, 0.5, 0, 0.5, 0))
    coord = np.array((1.5, 0.25))
    print(pysulfur.affine_transform(coord, affine))


def test_gasket():
    t1 = pysulfur.Transform(
        variations.Variation.as_weights(
            {
                variations.variation_fisheye.name: 0.25,
                variations.variation_bent.name: 0.125,
            }
        ),
        variations.Variation.as_params({}),
        np.array((0.75, 0, 0, 0, 0.75, 0)),
        1.0 / 3,
        0,
    )
    t2 = pysulfur.Transform(
        variations.Variation.as_weights(
            {
                variations.variation_diamond.name: 3,
                variations.variation_exponential.name: 1,
            }
        ),
        variations.Variation.as_params({}),
        np.array((0.75, 0, 0.25, 0, 0.75, 0)),
        1.0 / 3,
        1,
    )
    t3 = pysulfur.Transform(
        variations.Variation.as_weights(
            {
                variations.variation_pdj.name: 1,
            }
        ),
        variations.Variation.as_params(
            {
                variations.variation_pdj.name: [1, 2, 0.5, -1.2],
            }
        ),
        np.array((0.8, 0, 0, 0, 0.8, 0.34)),
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
        np.array((0.5, 0, 0.5, 0, 0.5, 0.5)),
    )
    seeds = list(range(1, 13))
    w, h = 400, 400
    array = flame.plot((w, h, 4), seeds, 20000)
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
