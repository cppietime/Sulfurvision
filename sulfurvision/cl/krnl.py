from os import path

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
import pyopencl.tools as cltools

from sulfurvision import pysulfur, variations

cl_types = {}
transform_type_key = 'transform_t'
particle_type_key = 'particle_t'

def transform_to_cl(transforms, q):
    if transform_type_key not in cl_types:
        raise Exception('Types have not yet been defined')
    host_transform_type = cl_types[transform_type_key]
    host_transforms = np.empty(len(transforms), host_transform_type)
    for i, transform in enumerate(transforms):
        host_transforms[i]['weights'] = transform.weights
        host_transforms[i]['params'] = transform.params
        host_transforms[i]['affine'] = transform.affine
        host_transforms[i]['probability'] = transform.probability
        host_transforms[i]['color'] = transform.color
        host_transforms[i]['color_speed'] = transform.color_speed
    return clarray.to_device(q, host_transforms)

def register_type(device, name, nptype):
    host_type, dev_type = cltools.match_dtype_to_c_struct(device, name, nptype)
    host_type = cltools.get_or_register_dtype(name, host_type)
    cl_types[name] = host_type
    return dev_type

def define_types(device):
    np_transform = np.dtype([
        ('weights', f'{len(variations.Variation.variations)}f4'),
        ('params', f'{variations.Variation.param_counter}f4'),
        ('affine', '6f4'),
        ('probability', 'f4'),
        ('color', 'f4'),
        ('color_speed', 'f4')
        ])
    dev_transform = register_type(device, transform_type_key, np_transform)

    np_particle = np.dtype([
        ('xy', clarray.vec.float2),
        ('seed', 'u4'),
        ('color', 'f4')
    ])
    dev_particle = register_type(device, particle_type_key, np_particle)

    srcs = [
        dev_transform,
        dev_particle
    ]
    return '\n'.join(srcs)
    

def combine_source(device):
    folder = path.split(__file__)[0]
    srcs = []
    # #defines
    defs_file = path.join(folder, 'defines.cl')
    with open(defs_file, 'r') as file:
        srcs.append(file.read())
    # Typedefs
    typedefs = define_types(device)
    srcs.append(typedefs)
    # Util functions
    util_file = path.join(folder, 'util.cl')
    with open(util_file, 'r') as file:
        srcs.append(file.read())
    # Variations
    variations_file = path.join(folder, 'variations.cl')
    with open(variations_file, 'r') as file:
        srcs.append(file.read())
    # Main kernel
    kernel_file = path.join(folder, 'kernel.cl')
    with open(kernel_file, 'r') as file:
        kernel_src = file.read()
    variations_srcs = []
    for i, variation in enumerate(variations.Variation.variations):
        if f'VARIATION({variation.name[len("variation_"):]})' not in srcs[-1]:
            continue
        variations_srcs.append(f'''            if (fabs(transform->weights[{i}]) > EPSILON) new_xy += transform->weights[{i}] * {variation.name}(xyrt, &seed, &(transform->params[{variation.params_base}]), transform->affine, transform->weights[{i}]);''')
    kernel_src = kernel_src.replace('@@VARIATIONS@@', '\n'.join(variations_srcs))
    srcs.append(kernel_src)
    src = '\n'.join(srcs)
    return src

def build_kernel(ctx, device):
    src = combine_source(device)
    program = cl.Program(ctx, src)
    return program.build()
