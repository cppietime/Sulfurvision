import typing

import numpy as np

"""A 2-dimensional affine transform of form A B C D E F.
Applying this transform to (x, y) is equivalent to:
(A*x + B*y + C, D*x + E*y + F)
"""
AffineTransform = tuple[float, float, float, float, float, float]
IdentityAffine = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
# A list of all the parameters used by all variations in order
ParamsList = list[float]
# A 2-D coordinate in continuous real space
Coord = tuple[float, float]
# A variation transform function that may access the current coordinate, affine transform, and its own parameters
VariationFunc = typing.Callable[
    [Coord, AffineTransform, ParamsList, int], tuple[Coord, int]
]
# Union of different possible color types
Color = np.ndarray[tuple[int], np.dtype[np.float64]]
# Maps linear color index to a color value
Colorizer = typing.Callable[[float], Color]
ImageGrid = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
