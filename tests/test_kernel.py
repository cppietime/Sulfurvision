import numpy as np
from PIL import Image
import pyopencl as cl
import pyopencl.array as clarray
from pyopencl import cltypes

from sulfurvision import prng, pysulfur, variations
from sulfurvision.cl import bootstrap, krnl

def rand_seed(base):
    base, x = prng.rand_uniform(base)
    base, y = prng.rand_uniform(base)
    base, color = prng.rand_uniform(base)
    return (cltypes.make_float2(x, y), base, color)

def test_flame(ctx, device, q):
    krnl.define_types(device)
    w = h = 200
    # TODO: test each transform (at least one is broken)
    transforms = [
        pysulfur.Transform(
            variations.Variation.as_weights({
                variations.variation_linear.name: 1
            }),
            variations.Variation.as_params({}),
            np.array([0.5, 0, 0, 0, 0.5, 0]),
            1/3,
            0,
        ),
        pysulfur.Transform(
            variations.Variation.as_weights({
                variations.variation_linear.name: 1,
                variations.variation_polar.name: 0,
            }),
            variations.Variation.as_params({}),
            np.array([0.5, 0, 0.5, 0, 0.5, 0]),
            1/3,
            1,
        ),
        pysulfur.Transform(
            variations.Variation.as_weights({
                variations.variation_linear.name: 0.5,
                variations.variation_horseshoe.name: 0.5,
            }),
            variations.Variation.as_params({}),
            np.array([0.5, 0, 0, 0, 0.75, -0.25]),
            1/3,
            2,
        ),
    ]
    dev_transforms = krnl.transform_to_cl(transforms, q)
    array = clarray.zeros(q, (h, w, 4), np.uint32)
    palette = clarray.to_device(q, np.array([
        cltypes.make_float4(255, 0, 0, 1),
        cltypes.make_float4(0, 255, 0, 1),
        cltypes.make_float4(0, 0, 255, 1),
    ], cltypes.float4))
    camera = clarray.to_device(q, np.array([w / 2, 0, w / 2, 0, h / 2, h / 2], np.float32))
    img_size = cltypes.make_uint2(w, h)
    n_seeds = 2000
    particles = clarray.to_device(q, np.array([rand_seed(prng.lcg32_skip(12345, i << 8)) for i in range(n_seeds)], krnl.cl_types[krnl.particle_type_key]))
    print(particles[0])
    prg = krnl.build_kernel(ctx, device)
    print(prg.kernel_names)
    krn = prg.flame_kernel
    krn(q, (n_seeds,), None,
        particles.data,
        array.data,
        dev_transforms.data,
        palette.data,
        camera.data,
        np.uint32(123456),
        np.uint32(100),
        np.uint32(10),
        img_size.data,
        np.uint32(3),
        np.uint32(3)
        )
    result = array.get()
    img = Image.new('RGB', (w, h))
    for y in range(h):
        for x in range(w):
            pix = result[x, y]
            img.putpixel((x, y), tuple(pix[:3]))
    img.save('test_cl.png')

def main():
    ctx = bootstrap.create_ctx()
    device = bootstrap.pick_device(ctx)
    q = cl.CommandQueue(ctx, device)
    print('Created queue')
    test_flame(ctx, device, q)

if __name__ == '__main__':
    main()
