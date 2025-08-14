import numpy as np
from PIL import Image

from sulfurvision import pysulfur, util

frame0_json = """
[
{
"weights": {"variation_julia": 1, "variation_polar": 2},
"params": {},
"affine": [1, 0, 0, 0, 1, 0],
"probability": 1,
"color": 0,
"color_speed": 0.5
},
{
"weights": {"variation_pdj": 1, "variation_fisheye": 2},
"params": {"variation_pdj": [1, -0.5, 1.5, 0.7]},
"affine": [0.5, 0, 0.5, 0, 0.5, 0],
"probability": 1,
"color": 1,
"color_speed": 0.5
}
]
"""

frame1_json = """
[
{
"weights": {"variation_julia": 3, "variation_polar": 1},
"params": {},
"affine": [1, 0, 0, 0, 0.75, -0.2],
"probability": 1,
"color": 0,
"color_speed": 0.5
},
{
"weights": {"variation_pdj": 1, "variation_polar": 0.2},
"params": {"variation_pdj": [1, -0.2, 0.5, 0.7]},
"affine": [0, 1, 0, 1, 0, 0],
"probability": 0.5,
"color": 1,
"color_speed": 0.5
}
]
"""

def test_frame0():
    frame0_ts = pysulfur.Transform.read_json(frame0_json)
    assert isinstance(frame0_ts, list)
    red = np.array([1, 0, 0, 1])
    green = np.array([0, 1, 0, 1])
    blue = np.array([0, 0, 1, 1])
    wheel = [red, green, blue, red]
    flame = pysulfur.Flame(
        frame0_ts,
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
    img.save("test_frame0.png")

def test_anim():
    frame0_ts = pysulfur.Transform.read_json(frame0_json)
    frame1_ts = pysulfur.Transform.read_json(frame1_json)
    assert(isinstance(frame0_ts, list))
    assert(isinstance(frame1_ts, list))
    red = np.array([1, 0, 0, 1])
    green = np.array([0, 1, 0, 1])
    blue = np.array([0, 0, 1, 1])
    wheel = [red, green, blue, red]
    flame = pysulfur.Flame(
        frame0_ts,
        lambda x: util.lerp(wheel[int(x)], wheel[int(x) + 1], x - int(x)),
        np.array((0.5, 0, 0.5, 0, 0.5, 0.5)),
    )
    seeds = list(range(1, 13))
    num_frames = 10
    w, h = 200, 200
    for n_frame in range(num_frames):
        z = n_frame / (num_frames - 1)
        transforms = [pysulfur.Transform.lerp(a, b, z / 20) for (a, b) in zip(frame0_ts, frame1_ts)]
        flame.transforms = transforms
        flame.update_total_weight()
        array = flame.plot((w, h, 4), seeds, 20000)
        img = Image.new("RGB", (w, h))
        for y in range(h):
            for x in range(w):
                if abs(array[x, y, -1]) > 1e-9:
                    img.putpixel(
                        (x, y),
                        tuple((array[x, y, :-1] * 255 / array[x, y, -1]).astype(int)),
                    )
        filename = f'test_anim{n_frame}.png'
        img.save(f"test_anim{n_frame}.png")
        print(f'Saved {filename}...')

def main():
    test_anim()

if __name__ == '__main__':
    main()
