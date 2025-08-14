from os import path

import pyopencl as cl

def build_kernel(ctx):
    with open(path.join(path.split(__file__)[0], 'test_kernel.cl')) as file:
        src = file.read()
    program = cl.Program(ctx, src)
    return program.build()
