import numpy as np
from PIL import Image
import pyopencl as cl
import pyopencl.array as clarray
from pyopencl import cltypes

from sulfurvision import prng
from sulfurvision.cl import bootstrap, krnl

def rand_particle(seed):
    seed, x = prng.rand_uniform(seed)
    seed, y = prng.rand_uniform(seed)
    seed, color = prng.rand_uniform(seed)
    return (cltypes.make_float2(x, y), seed, color)

class Renderer:
    def __init__(self, w, h, supersample, n_particles, n_colors, n_variations):
        
        self.ctx = bootstrap.create_ctx()
        self.device = bootstrap.pick_device(self.ctx)
        self.queue = cl.CommandQueue(self.ctx, self.device)
        self.program = krnl.build_kernel(self.ctx, self.device)
        self.kernels = [
            self.program.flame_kernel,
            self.program.downsample_kernel,
            self.program.rowmax_kernel,
            self.program.tonemap_kernel
        ]
        self.size = (w, h)
        self.supersample = supersample
        self.n_particles = n_particles
        self.n_colors = n_colors
        self.n_variations = n_variations
        self.pixel_array = clarray.zeros(self.queue, w * h * 4, np.uint32)
        self.histogram = clarray.zeros(self.queue, w * h * 4 * supersample * supersample, np.uint32)
        self.particles = clarray.empty(self.queue, (n_particles,), krnl.cl_types[krnl.particle_type_key])
        self.palette = clarray.empty(self.queue, n_colors, cltypes.float4)
        self.camera = clarray.zeros(self.queue, 6, np.float32)
        self.variations = clarray.empty(self.queue, (n_variations,), krnl.cl_types[krnl.transform_type_key])
    
    def render(self, camera, transforms, palette, iters, skip, vibrancy=1, gamma=0.8, brightness=20):
        self.particles.set(np.array([
            rand_particle(i) for i in range(self.n_particles)
            ], dtype=krnl.cl_types[krnl.particle_type_key]))
        self.histogram.fill(0)
        self.pixel_array.fill(0)
        self.palette.set(np.asarray([cltypes.make_float4(*color) for color in palette], cltypes.float4))
        print(type(self.palette))
        print(self.palette.dtype)
        print(self.palette.shape)
        self.camera.set(np.asarray(camera, np.float32))
        krnl.transform_into_cl(transforms, self.variations)
        self.kernels[0](self.queue, (self.n_particles,), None,
                        self.particles.data,
                        self.histogram.data,
                        self.variations.data,
                        self.palette.data,
                        self.camera.data,
                        np.uint32(iters),
                        np.uint32(skip),
                        cltypes.make_uint2(*self.size),
                        np.uint32(self.n_variations),
                        np.uint32(self.n_colors),
                        np.uint32(self.supersample)
                        ).wait()
        if self.supersample > 1:
            self.kernels[1](self.queue, (self.size[0] * self.size[1],), None,
                            self.histogram.data,
                            self.pixel_array.data,
                            cltypes.make_uint2(*self.size),
                            np.uint32(self.supersample)
                            ).wait()
        else:
            self.pixel_array.set(self.histogram.get())
        self.kernels[2](self.queue, (self.size[1],), None,
                        self.pixel_array.data,
                        self.histogram.data,
                        cltypes.make_uint2(*self.size)
                        ).wait()
        maxima = self.histogram.get()[:self.size[1]]
        maximum = maxima.max()
        self.kernels[3](self.queue, (self.size[0] * self.size[1],), None,
                        self.pixel_array.data,
                        cltypes.make_uint2(*self.size),
                        np.float32(brightness),
                        np.float32(gamma),
                        np.float32(vibrancy),
                        np.uint32(maximum),
                        np.uint32(1)
                        ).wait()
        imgdata = self.pixel_array.get()
        return Image.fromarray(imgdata.reshape(*self.size, 4)[:,:,:3].astype(np.uint8))

