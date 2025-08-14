import numpy as np
from PIL import Image
import pyopencl as cl
import pyopencl.array as clarray

from sulfurvision.cl import bootstrap, krnl

def test_rand(ctx, q):
    w = h = 100
    array = clarray.empty(q, (w, h), np.float32)
    prg = krnl.build_kernel(ctx)
    print(prg.kernel_names)
    krn = prg.test_kernel
    krn(q, (w * h,), None, array.data, np.float32(w), np.float32(h))
    result = array.get()
    img = Image.new('L', (w, h))
    for y in range(h):
        for x in range(w):
            pix = int(abs(result[x, y]) * 255)
            img.putpixel((x, y), pix)
    img.save('test_cl.png')

def main():
    ctx = bootstrap.create_ctx()
    q = cl.CommandQueue(ctx)
    test_rand(ctx, q)

if __name__ == '__main__':
    main()
