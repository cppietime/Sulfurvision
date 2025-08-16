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

def test_transform(w, h, supersample, q, flame_kernel, pool_kernel, rowmax_kernel, tone_kernel, transform_name, histogram, array):
    all_weights = np.full(len(variations.Variation.variations), 0, np.float32)
    all_weights[variations.Variation.variations_map[variations.variation_linear.name]] = 1
    all_weights[variations.Variation.variations_map[transform_name]] = 1
    all_weights /= all_weights.sum()
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
            }),
            variations.Variation.as_params({}),
            np.array([0.5, 0, 0.5, 0, 0.5, 0]),
            1/3,
            1,
        ),
        pysulfur.Transform(
            all_weights,
            variations.Variation.as_params({
                variations.variation_juliaN.name: [2, 0.75],
                variations.variation_juliaScope.name: [2, 0.75],
                variations.variation_ngon.name: [2, 4, 4, 0.5],
                variations.variation_rectangles.name: [1, -0.3],
                variations.variation_radialBlur.name: [0.7],
            }),
            np.array([0.5, 0, 0.15, 0, 0.75, -0.25]),
            1/3,
            2,
        ),
    ]
    dev_transforms = krnl.transform_to_cl(transforms, q)
    palette = clarray.to_device(q, np.array([
        cltypes.make_float4(255, 0, 0, 1),
        cltypes.make_float4(0, 255, 0, 1),
        cltypes.make_float4(0, 0, 255, 1),
    ], cltypes.float4))
    camera = clarray.to_device(q, np.array([w * supersample / 2, 0, w * supersample  / 2, 0, h * supersample  / 2, h * supersample  / 2], np.float32))
    img_size = cltypes.make_uint2(w, h)
    n_seeds = 2000
    particles = clarray.to_device(q, np.array([rand_seed(prng.lcg32_skip(12345, i << 8)) for i in range(n_seeds)], krnl.cl_types[krnl.particle_type_key]))
    histogram.fill(np.float32(0))
    array.fill(np.float32(0))
    flame_kernel(q, (n_seeds,), None,
        particles.data,
        histogram.data,
        dev_transforms.data,
        palette.data,
        camera.data,
        np.uint32(123456),
        np.uint32(100),
        np.uint32(10),
        img_size.data,
        np.uint32(3),
        np.uint32(3),
        np.uint32(supersample)
        ).wait()
    pool_kernel(q, (n_seeds,), None,
        histogram.data,
        array.data,
        img_size.data,
        np.uint32(supersample)).wait()
    rowmax_kernel(q, (n_seeds,), None,
        array.data,
        histogram.data,
        img_size.data).wait()
    rowmaxima = histogram.get()[:h]
    maximum = rowmaxima.max()
    print(f'Max alpha: {maximum}')
    tone_kernel(q, (n_seeds,), None,
        array.data,
        img_size.data).wait()
    result = array.get().reshape((w, h, 4))
    img = Image.new('RGB', (w, h))
    for y in range(h):
        for x in range(w):
            pix = result[x, y]
            img.putpixel((x, y), tuple(pix[:3]))
    img.save(f'{transform_name}.png')

def test_flame(ctx, device, q):
    krnl.define_types(device)
    w = h = 200
    supersample = 4
    histogram = clarray.zeros(q, (4 * w * h * (supersample ** 2)), np.uint32)
    array = clarray.zeros(q, (h * w * 4,), np.uint32)
    prog = krnl.build_kernel(ctx, device)
    flame = prog.flame_kernel
    pool = prog.downsample_kernel
    rowmax = prog.rowmax_kernel
    tone = prog.tonemap_kernel
    for variation in variations.Variation.variations:
        test_transform(w, h, supersample, q, flame, pool, rowmax, tone, variation.name, histogram, array)
        break

def main():
    ctx = bootstrap.create_ctx()
    device = bootstrap.pick_device(ctx)
    q = cl.CommandQueue(ctx, device)
    print('Created queue')
    test_flame(ctx, device, q)

if __name__ == '__main__':
    main()
