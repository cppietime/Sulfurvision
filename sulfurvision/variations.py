import dataclasses
import typing

import numpy as np

from sulfurvision import types


@dataclasses.dataclass
class Variation:
    """Metadata about a variation that a transform may use."""

    function: types.VariationFunc
    num_params: int
    params_base: int = dataclasses.field(init=False, default=0)

    variations: typing.ClassVar[list["Variation"]] = []
    param_counter: typing.ClassVar[int] = 0

    def __post_init__(self):
        self.params_base = Variation.param_counter
        Variation.param_counter += self.num_params
        Variation.variations.append(self)

    def __call__(
        self,
        coord: types.Coord,
        affine: types.AffineTransform,
        params: types.ParamsList,
        seed: int,
    ) -> tuple[types.Coord, int]:
        return self.function(
            coord,
            affine,
            params[self.params_base : self.params_base + self.num_params],
            seed,
        )


def WrapVariation(
    num_params: int = 0,
) -> typing.Callable[[types.VariationFunc], Variation]:
    def inner(varfunc: types.VariationFunc) -> Variation:
        return Variation(varfunc, num_params)

    return inner


# All transforms declared below
@WrapVariation()
def variation_linear(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    return xy, seed


@WrapVariation()
def variation_sinusoidal(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    return (np.sin(xy[0]), np.sin(xy[1])), seed


@WrapVariation()
def variation_spherical(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    r_recip = 1 / (xy[0] ** 2 + xy[1] ** 2)
    return (xy[0] * r_recip, xy[1] * r_recip), seed


@WrapVariation()
def variation_swirl(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    rsq = xy[0] ** 2 + xy[1] ** 2
    sin = np.sin(rsq)
    cos = np.cos(rsq)
    return (xy[0] * sin - xy[1] * cos, xy[0] * cos + xy[1] * sin), seed


@WrapVariation()
def variation_horseshoe(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    r_recip = 1 / np.hypot(*xy)
    return (
        (xy[0] - xy[1]) * (xy[0] + xy[1]) * r_recip,
        2 * r_recip * xy[0] * xy[1],
    ), seed


@WrapVariation()
def variation_polar(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    theta = np.atan2(*xy)
    r = np.hypot(*xy)
    return (theta / np.pi, r - 1), seed


@WrapVariation()
def variation_handkerchief(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    r = np.hypot(*xy)
    theta = np.atan2(*xy)
    return (r * np.sin(theta + r), r * np.cos(theta - r)), seed


@WrapVariation()
def variation_heart(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    r = np.hypot(*xy)
    theta = np.atan2(*xy)
    return (r * np.sin(r * theta), -r * np.cos(r * theta)), seed


# End of transformation definitions
