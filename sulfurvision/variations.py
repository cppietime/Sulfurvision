import dataclasses
import typing

import numpy as np

from sulfurvision import prng, types


@dataclasses.dataclass
class Variation:
    """Metadata about a variation that a transform may use."""

    function: types.VariationFunc
    num_params: int
    name: str
    params_base: int = dataclasses.field(init=False, default=0)

    variations: typing.ClassVar[list["Variation"]] = []
    variations_map: typing.ClassVar[dict[str, int]] = {}
    param_counter: typing.ClassVar[int] = 0

    def __post_init__(self):
        self.params_base = Variation.param_counter
        Variation.param_counter += self.num_params
        Variation.variations_map[self.name] = len(Variation.variations)
        Variation.variations.append(self)
        assert Variation.variations[Variation.variations_map[self.name]] is self

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

    @staticmethod
    def as_weights(weights_dict: dict[str, float]) -> types.ParamsList:
        weights = np.zeros(len(Variation.variations))
        total = sum(weights_dict.values())
        if total == 0:
            return weights
        for name, weight in weights_dict.items():
            weights[Variation.variations_map[name]] = weight / total
        return weights

    @staticmethod
    def as_params(params_dict: dict[str, list[float]]) -> types.ParamsList:
        params = np.zeros(Variation.param_counter)
        for name, vals in params_dict.items():
            base = Variation.variations[Variation.variations_map[name]].params_base
            for i, val in enumerate(vals):
                params[base + i] = val
        return params


def WrapVariation(
    num_params: int = 0,
) -> typing.Callable[[types.VariationFunc], Variation]:
    def inner(varfunc: types.VariationFunc) -> Variation:
        return Variation(varfunc, num_params, varfunc.__name__)

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
    return np.sin(xy), seed


@WrapVariation()
def variation_spherical(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    r_recip = 1 / (xy ** 2).sum()
    return xy * r_recip, seed


@WrapVariation()
def variation_swirl(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    rsq = (xy ** 2).sum()
    sin = np.sin(rsq)
    cos = np.cos(rsq)
    return np.array((xy[0] * sin - xy[1] * cos, xy[0] * cos + xy[1] * sin)), seed


@WrapVariation()
def variation_horseshoe(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    r_recip = 1 / np.hypot(*xy)
    return np.array((
        (xy[0] - xy[1]) * (xy[0] + xy[1]) * r_recip,
        2 * r_recip * xy[0] * xy[1],
    )), seed


@WrapVariation()
def variation_polar(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    theta = np.atan2(*xy)
    r = np.hypot(*xy)
    return np.array((theta / np.pi, r - 1)), seed


@WrapVariation()
def variation_handkerchief(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    r = np.hypot(*xy)
    theta = np.atan2(*xy)
    return r * np.array((np.sin(theta + r), np.cos(theta - r))), seed


@WrapVariation()
def variation_heart(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    r = np.hypot(*xy)
    theta = np.atan2(*xy)
    return r * np.array((np.sin(r * theta), -np.cos(r * theta))), seed


@WrapVariation()
def variation_disc(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    r = np.hypot(*xy) * np.pi
    theta = np.atan2(*xy) / np.pi
    return theta * np.array((np.sin(r), np.cos(r))), seed


@WrapVariation()
def variation_spiral(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    r = np.hypot(*xy)
    theta = np.atan2(*xy)
    return np.array(((np.cos(theta) + np.sin(r)) / r, (np.sin(theta) - np.cos(r)) / 2)), seed


@WrapVariation()
def variaton_hyperbolic(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    r = np.hypot(*xy)
    theta = np.atan2(*xy)
    return np.array((np.sin(theta) / r, r * np.cos(theta))), seed


@WrapVariation()
def variation_diamond(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    r = np.hypot(*xy)
    theta = np.atan2(*xy)
    return np.array((np.sin(theta) * np.cos(r), np.cos(theta) * np.sin(r))), seed


@WrapVariation()
def variation_ex(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    r = np.hypot(*xy)
    theta = np.atan2(*xy)
    p0 = np.sin(theta + r) ** 3
    p1 = np.cos(theta - r) ** 3
    return r * np.array(((p0 + p1), (p0 - p1))), seed


@WrapVariation()
def variation_julia(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    sqrt_r = np.sqrt(np.hypot(*xy))
    h_theta = np.atan2(*xy) / 2
    new_seed = prng.rand_u32(seed)
    w = np.pi if (new_seed & 1) else 0
    return sqrt_r * np.array((np.cos(h_theta + w),  np.sin(h_theta + w))), new_seed


@WrapVariation()
def variation_bent(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    x, y = xy
    if x >= 0 and y >= 0:
        return xy, seed
    elif y >= 0:
        return np.array((x * 2, y)), seed
    elif x >= 0:
        return np.array((x, y / 2)), seed
    else:
        return np.array((x * 2, y / 2)), seed


@WrapVariation()
def variation_waves(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    x, y = xy
    _, b, c, _, e, f = affine
    if abs(c) < 1e-9:
        c = 1
    if abs(f) < 1e-9:
        f = 1
    return np.array((x + b * np.sin(y / c**2), y + e * np.sin(x / f**2))), seed


@WrapVariation()
def variation_fisheye(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    r = np.hypot(*xy) + 1
    return 2 / r * xy[::-1], seed


@WrapVariation()
def variation_popcorn(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    x, y = xy
    _, _, c, _, _, f = affine
    return np.array((x + c * np.sin(np.tan(3 * y), y + f * np.sin(np.tan(3 * x))))), seed


@WrapVariation()
def variation_exponential(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    exp = np.exp(xy[0] - 1)
    piy = np.pi * xy[1]
    return exp * np.array((np.cos(piy), np.sin(piy))), seed


@WrapVariation()
def variation_power(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    theta = np.atan2(*xy)
    sinth = np.sin(theta)
    r = np.hypot(*xy) ** sinth
    return r * np.array((np.cos(theta), sinth)), seed


@WrapVariation()
def variation_cosine(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    pix = np.pi * xy[0]
    return np.array((np.cos(pix) * np.cosh(xy[1]), -np.sin(pix) * np.sinh(xy[1]))), seed


@WrapVariation()
def variation_rings(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    r = np.hypot(*xy)
    c = affine[2]
    if abs(c) < 1e-9:
        c = 1
    theta = np.atan2(*xy)
    factor = ((r + c * c) % (2 * c * c)) - c * c + r * (1 - c * c)
    return factor * np.array((np.cos(theta), np.sin(theta))), seed


@WrapVariation()
def variation_fan(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    _, _, c, _, _, f = affine
    t = np.pi * c * c
    theta = np.atan2(*xy)
    r = np.hypot(*xy)
    if (theta + f) % t >= t / 2:
        arg = theta - t / 2
    else:
        arg = theta + t / 2
    return r * np.array((np.cos(arg), np.sin(arg))), seed


@WrapVariation(3)
def variation_blob(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    r = np.hypot(*xy)
    theta = np.atan2(*xy)
    factor = r * (
        params[1] + (params[0] - params[1]) / 2 * (np.sin(params[3] * theta) + 1)
    )
    return factor * np.array((np.cos(theta), np.sin(theta))), seed


@WrapVariation(4)
def variation_pdj(
    xy: types.Coord, affine: types.AffineTransform, params: types.ParamsList, seed: int
):
    return np.array((
        np.sin(params[0] * xy[1]) - np.cos(params[1] * xy[0]),
        np.sin(params[2] * xy[0]) - np.cos(params[3] * xy[1]),
    )), seed


# End of transformation definitions
