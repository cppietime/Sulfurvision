import dataclasses
import typing

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
from PIL import Image
from pyopencl import cltypes

from sulfurvision import prng, pysulfur, types
from sulfurvision.cl import bootstrap, krnl


def rand_particle(seed: int) -> tuple[cltypes.float2, int, float]:
    seed, x = prng.rand_uniform(seed)
    seed, y = prng.rand_uniform(seed)
    seed, color = prng.rand_uniform(seed)
    return (cltypes.make_float2(x, y), seed, color)


@dataclasses.dataclass
class RenderFrame:
    """Similar to pysulfur.Flame.
    Holds information for rendering one frame.
    """

    transforms: typing.Sequence[pysulfur.Transform]
    palette: typing.Sequence[types.Color]
    camera: types.AffineTransform
    time: float

    def __mul__(self, other) -> "RenderFrame":
        return RenderFrame(
            np.asarray(self.transforms, dtype=np.float64) * other,
            np.asarray(self.palette, dtype=np.float64) * other,
            np.asarray(self.camera, dtype=np.float64) * other,
            self.time * other,
        )

    def __add__(self, other) -> "RenderFrame":
        assert len(self.transforms) == len(other.transforms)
        assert len(self.palette) == len(other.palette)
        return RenderFrame(
            np.asarray(self.transforms, dtype=np.float64)
            + np.asarray(other.transforms, dtype=np.float64),
            np.asarray(self.palette, dtype=np.float64)
            + np.asarray(other.palette, dtype=np.float64),
            np.asarray(self.camera, dtype=np.float64)
            + np.asarray(other.camera, dtype=np.float64),
            self.time + other.time,
        )


class Renderer:
    """Utility class for rendering flames."""

    _ctx = None
    _device = None
    _queue = None
    _program = None
    _kernels = None

    @classmethod
    def _init_cl(cls):
        """Helper class method to initialize global CL objects."""
        if cls._ctx is None:
            cls._ctx = bootstrap.create_ctx()
        if cls._device is None:
            cls._device = bootstrap.pick_device(cls._ctx)
        if cls._queue is None:
            cls._queue = cl.CommandQueue(cls._ctx, cls._device)
        if cls._program is None:
            cls._program = krnl.build_kernel(cls._ctx, cls._device)
        if cls._kernels is None:
            cls._kernels = [
                cls._program.flame_kernel,
                cls._program.downsample_kernel,
                cls._program.rowmax_kernel,
                cls._program.tonemap_kernel,
            ]

    def __init__(
        self,
        w: int,
        h: int,
        supersample: int,
        n_particles: int,
        n_colors: int,
        n_variations: int,
        seed: int = 12345,
    ):
        Renderer._init_cl()
        self.w = w
        self.h = h
        self.img_size = cltypes.make_uint2(w, h)
        self.supersample = supersample
        self.n_particles = n_particles
        self.n_colors = n_colors
        self.n_variations = n_variations
        self.seed = seed
        self.pixel_array = clarray.zeros(Renderer._queue, w * h * 4, np.uint32)
        self.histogram = clarray.zeros(
            Renderer._queue, w * h * 4 * supersample * supersample, np.uint32
        )
        self.row_ctr = clarray.zeros(Renderer._queue, h, np.uint32)
        self.particles = clarray.empty(
            Renderer._queue, (n_particles,), krnl.cl_types[krnl.particle_type_key]
        )
        self.palette = clarray.empty(Renderer._queue, n_colors, cltypes.float4)
        self.camera = clarray.zeros(Renderer._queue, 6, np.float32)
        self.variations = clarray.empty(
            Renderer._queue, (n_variations,), krnl.cl_types[krnl.transform_type_key]
        )

    def update_to_match(
        self,
        w: int,
        h: int,
        supersample: int,
        n_particles: int,
        n_colors: int,
        n_variations: int,
    ) -> None:
        if (
            self.w == w
            and self.h == h
            and self.supersample == supersample
            and self.n_particles == n_particles
            and self.n_colors == n_colors
            and self.n_variations == n_variations
        ):
            return
        self.w = w
        self.h = h
        self.supersample = supersample
        self.n_particles = n_particles
        self.n_colors = n_colors
        self.n_variations = n_variations
        self.img_size = cltypes.make_uint2(w, h)
        self.pixel_array = clarray.zeros(Renderer._queue, w * h * 4, np.uint32)
        self.histogram = clarray.zeros(
            Renderer._queue, w * h * 4 * supersample * supersample, np.uint32
        )
        self.row_ctr = clarray.zeros(Renderer._queue, h, np.uint32)
        self.particles = clarray.empty(
            Renderer._queue, (n_particles,), krnl.cl_types[krnl.particle_type_key]
        )
        self.palette = clarray.empty(Renderer._queue, n_colors, cltypes.float4)
        self.variations = clarray.empty(
            Renderer._queue, (n_variations,), krnl.cl_types[krnl.transform_type_key]
        )

    def chaos_game(
        self,
        camera: types.AffineTransform,
        transforms: typing.Sequence[pysulfur.Transform],
        palette: types.Palette,
        iters: int,
        skip: int,
    ):
        """Run the chaos game, and do nothing else that is not necessary for it."""
        self.palette.set(
            np.asarray(
                [cltypes.make_float4(*color) for color in palette], cltypes.float4
            )
        )
        self.camera.set(np.asarray(camera, np.float32))
        krnl.transform_into_cl(transforms, self.variations)
        Renderer._kernels[0](
            Renderer._queue,
            (self.n_particles,),
            None,
            self.particles.data,
            self.histogram.data,
            self.variations.data,
            self.palette.data,
            self.camera.data,
            np.uint32(iters),
            np.uint32(skip),
            self.img_size,
            np.uint32(self.n_variations),
            np.uint32(self.n_colors),
            np.uint32(self.supersample),
        ).wait()

    def image(
        self, vibrancy: float = 1, gamma: float = 0.8, brightness: float = 20
    ) -> Image.Image:
        """Return an image from the current chaos game state, using the following steps:
        - Downsample, if necessary,
        - Calculate the maximum alpha,
        - Perform tonemapping,
        - Return a PIL Image RGB object
        """
        self.pixel_array.fill(0)
        if self.supersample > 1:
            Renderer._kernels[1](
                Renderer._queue,
                (self.w * self.h,),
                None,
                self.histogram.data,
                self.pixel_array.data,
                self.img_size,
                np.uint32(self.supersample),
            ).wait()
        else:
            self.pixel_array.set(self.histogram.get())
        Renderer._kernels[2](
            Renderer._queue,
            (self.w,),
            None,
            self.pixel_array.data,
            self.row_ctr.data,
            self.img_size,
        ).wait()
        maximum = self.row_ctr.get().max()
        Renderer._kernels[3](
            Renderer._queue,
            (self.w * self.h,),
            None,
            self.pixel_array.data,
            self.img_size,
            np.float32(brightness),
            np.float32(gamma),
            np.float32(vibrancy),
            np.uint32(maximum),
            np.uint32(1),
        ).wait()
        imgdata = self.pixel_array.get()
        return Image.fromarray(
            imgdata.reshape(self.h, self.w, 4)[:, :, :3].astype(np.uint8)
        )

    def reset(self):
        """Fill the histogram with 0s"""
        self.histogram.fill(0)

    def randomize_particles(self):
        """Reset all particles to pseudo-random starting points"""
        self.particles.set(
            np.array(
                [
                    rand_particle(prng.lcg32_skip(self.seed, i << 8))
                    for i in range(self.n_particles)
                ],
                dtype=krnl.cl_types[krnl.particle_type_key],
            )
        )
        self.seed = prng.lcg32_skip(self.seed, (self.n_particles << 8) + 1)

    def render(
        self,
        camera: types.AffineTransform,
        transforms: typing.Sequence[pysulfur.Transform],
        palette: types.Palette,
        iters: int,
        skip: int,
        vibrancy: float = 1,
        gamma: float = 0.8,
        brightness: float = 20,
    ) -> Image.Image:
        """Perform a start-to-finish rendering job, returning a PIL RGB Image object."""
        self.reset()
        self.randomize_particles()
        self.chaos_game(camera, transforms, palette, iters, skip)
        return self.image(vibrancy, gamma, brightness)
