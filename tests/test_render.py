import numpy as np

from sulfurvision import pysulfur
from sulfurvision.cl import render

json_str = """
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
"affine": [0.5, 0, 0.45, 0, 0.5, 0],
"probability": 1,
"color": 1,
"color_speed": 0.5
},
{
"weights": {"variation_pdj": 1, "variation_fisheye": 2},
"params": {"variation_pdj": [1, -0.5, 1.5, 0.7]},
"affine": [0.5, 0, -0.05, 0, 0.5, 0.55],
"probability": 1,
"color": 2,
"color_speed": 0.5
}
]
"""
json_sier = """
[
{
"weights": {"variation_linear": 1},
"params": {},
"affine": [0.5, 0, 0, 0, 0.5, 0],
"probability": 1,
"color": 0,
"color_speed": 0.5
},
{
"weights": {"variation_linear": 1},
"params": {},
"affine": [0.5, 0, 0.5, 0, 0.5, 0],
"probability": 1,
"color": 1,
"color_speed": 0.5
},
{
"weights": {"variation_linear": 1},
"params": {},
"affine": [0.5, 0, 0, 0, 0.5, 0.5],
"probability": 1,
"color": 2,
"color_speed": 0.5
}
]
"""

def test_render():
    transforms = pysulfur.Transform.read_json(json_str)
    w = h = 8000
    sample = 2
    renderer = render.Renderer(w, h, sample, 10000, 3, len(transforms))
    camera = np.array([w * sample / 2, 0, w * sample / 2, 0, h * sample / 2, h * sample / 2])
    palette = np.array([
        [0, 255, 255, 1],
        [255, 0, 255, 1],
        [255, 255, 0, 1],
    ])
    img = renderer.render(camera, transforms, palette, 10000, 15, gamma=0.5, brightness=50, vibrancy=1)
    img.save('render.png')

def main():
    test_render()

if __name__ == '__main__':
    main()
