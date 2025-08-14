from os import path

import pyopencl as cl

def combine_source():
    folder = path.split(__file__)[0]
    srcs = []
    # #defines
    # Typedefs
    # Util functions
    # Variations
    variations_file = path.join(folder, 'variations.cl')
    with open(variations_file, 'r') as file:
        srcs.append(file.read())
    # Main kernel
    src = '\n'.join(srcs)
    return src

def build_kernel(ctx):
    src = combine_source()
    program = cl.Program(ctx, src)
    return program.build()
