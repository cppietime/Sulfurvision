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

def test_render(w, h, supersample, q, kernels, transforms, name, palette, histogram, array, n_seeds = 1000, gamma = 1, vibrancy = 0, brightness = 1, mode = 1):
    dev_transforms = krnl.transform_to_cl(transforms, q)
    camera = clarray.to_device(q, np.array([w * supersample / 2, 0, w * supersample  / 2, 0, h * supersample  / 2, h * supersample  / 2], np.float32))
    img_size = cltypes.make_uint2(w, h)
    particles = clarray.to_device(q, np.array([rand_seed(prng.lcg32_skip(12345, i << 8)) for i in range(n_seeds)], krnl.cl_types[krnl.particle_type_key]))
    histogram.fill(np.float32(0))
    array.fill(np.float32(0))
    flame_kernel, pool_kernel, rowmax_kernel, tone_kernel = kernels
    flame_kernel(q, (n_seeds,), None,
        particles.data,
        histogram.data,
        dev_transforms.data,
        palette.data,
        camera.data,
        np.uint32(1000),
        np.uint32(10),
        img_size.data,
        np.uint32(3),
        np.uint32(3),
        np.uint32(supersample)
        ).wait()
    if supersample > 1:
        pool_kernel(q, (n_seeds,), None,
            histogram.data,
            array.data,
            img_size.data,
            np.uint32(supersample)).wait()
    else:
        array.set(histogram.get())
    rowmax_kernel(q, (n_seeds,), None,
        array.data,
        histogram.data,
        img_size.data).wait()
    rowmaxima = histogram.get()[:h]
    maximum = rowmaxima.max()
    tone_kernel(q, (n_seeds,), None,
        array.data,
        img_size.data,
        np.float32(brightness),
        np.float32(gamma),
        np.float32(vibrancy),
        np.uint32(maximum),
        np.uint32(mode)).wait()
    result = array.get().reshape((w, h, 4))
    img = Image.fromarray(result[:,:,:3].astype(np.uint8), 'RGB')
    img.save(f'{name}.png')
    print(f'Saved {name}')

def test_transform(w, h, supersample, q, kernels, transform_name, histogram, array):
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
    palette = clarray.to_device(q, np.array([
        cltypes.make_float4(255, 0, 0, 1),
        cltypes.make_float4(0, 255, 0, 1),
        cltypes.make_float4(0, 0, 255, 1),
    ], cltypes.float4))
    test_render(w, h, supersample, q, kernels, transforms, transform_name, palette, histogram, array)

def test_brightness(w, h, supersample, q, kernels, histogram, array, brightness, gamma, vibrancy, mode, n_seeds = 1000):
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
                variations.variation_linear.name: 3,
                variations.variation_pdj.name: 0,
                variations.variation_diamond.name: 0
            }),
            variations.Variation.as_params({
                variations.variation_pdj.name: [0.5, 1.1, -0.4, 1]
            }),
            np.array([0.5, 0, 0.5, 0, 0.5, 0]),
            1/3,
            1,
        ),
        pysulfur.Transform(
            variations.Variation.as_weights({
                variations.variation_linear.name: 3,
                variations.variation_julia.name: 1,
                variations.variation_horseshoe.name: 0
            }),
            variations.Variation.as_params({}),
            np.array([0.5, 0, 0.2, 0, 0.75, -0.2]),
            1/3,
            2,
        ),
    ]
    palette = clarray.to_device(q, np.array([
        cltypes.make_float4(255, 255, 0, 1),
        cltypes.make_float4(0, 255, 255, 1),
        cltypes.make_float4(255, 0, 255, 1),
    ], cltypes.float4))
    name = f'B{brightness:.1f}G{gamma:.1f}V{vibrancy:.1f}M{mode:x}'
    test_render(w, h, supersample, q, kernels, transforms, name, palette, histogram, array, n_seeds=n_seeds, gamma=gamma, vibrancy=vibrancy, brightness=brightness, mode=mode)

def setup_flame_test(w, h, supersample, ctx, device):
    q = cl.CommandQueue(ctx, device)
    histogram = clarray.zeros(q, (4 * w * h * (supersample ** 2)), np.uint32)
    array = clarray.zeros(q, (h * w * 4,), np.uint32)
    prog = krnl.build_kernel(ctx, device)
    flame = prog.flame_kernel
    pool = prog.downsample_kernel
    rowmax = prog.rowmax_kernel
    tone = prog.tonemap_kernel
    kernels = (flame, pool, rowmax, tone)
    return histogram, array, kernels, q

def test_variations(ctx, device):
    w = h = 200
    supersample = 4
    histogram, array, kernels, q = setup_flame_test(w, h, supersample, ctx, device)
    for variation in variations.Variation.variations:
        test_transform(w, h, supersample, q, kernels, variation.name, histogram, array)

def test_brightnesses(ctx, device):
    w = h = 2000
    supersample = 2
    histogram, array, kernels, q = setup_flame_test(w, h, supersample, ctx, device)
    mode = 1
    for brightness in [4, 10, 20, 80]:
        test_brightness(w, h, supersample, q, kernels, histogram, array, brightness, 1, 1, mode, w)
    for gamma in [0.7, 0.9, 1, 1.5, 3]:
        for vibrancy in [0, 0.5, 1]:
            test_brightness(w, h, supersample, q, kernels, histogram, array, 20, gamma, vibrancy, mode, w)

def main():
    ctx = bootstrap.create_ctx()
    device = bootstrap.pick_device(ctx)
    krnl.define_types(device)
    test_brightnesses(ctx, device)

if __name__ == '__main__':
    main()
