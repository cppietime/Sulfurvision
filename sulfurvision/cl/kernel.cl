__kernel void flame_kernel(
    __global uint* image,
    __constant transform_t* transforms,
    __constant float4* palette,
    __constant float* camera,
    const uint base_seed,
    const uint n_itrs,
    const uint skip_itrs,
    const uint2 image_size,
    const uint n_transforms,
    const uint n_colors) {
        size_t id = get_global_id(0);
        size_t n_seeds = get_global_size(0);

        // Setup PRNG
        uint seed = lcg32_skip(base_seed, id << 8);
        float x = (seed & 0xff) / 256.0;
        float y = ((seed >> 8) & 0xff) / 256.0;
        float color = ((seed >> 16) & 0xff) / 256.0;
        color = fmod(color, n_colors);

        float2 xy = (float2)(x, y);

        for (uint i = 0; i < n_itrs; i++) {
            seed = lcg32(seed);
            // Select a transform based on weights
            float p = (float)seed / MASK32;
            uint t_choice;
            for (t_choice = 0; p >= 0; p -= transforms[t_choice++].probability);
            __constant transform_t* transform = transforms + t_choice - 1;

            // Lerp color
            color += (transform->color - color) * transform->color_speed;
            xy = affine_transform(transform->affine, xy);

            // Precalculate extra params
            float r = hypot(xy.x, xy.y);
            float theta = atan2(xy.y, xy.x);
            float4 xyrt = (float4)(xy, r, theta);
            float2 new_xy = 0;

@@VARIATIONS@@

            xy = new_xy;

            if (i >= skip_itrs) {
                float2 pixel = affine_transform(camera, xy);
                uint ux = (uint)pixel.x;
                uint uy = (uint)pixel.y;
                if (ux >= 0 && uy >= 0 && ux < image_size.x && uy < image_size.y) {
                    uint pixel_id = ux + uy * image_size.x;
                    __global uint* pixptr = image + pixel_id * 4;
                    uchar4 rgba = sample_palette(palette, color, n_colors);
                    atomic_add(pixptr + 0, rgba.r);
                    atomic_add(pixptr + 1, rgba.g);
                    atomic_add(pixptr + 2, rgba.b);
                    atomic_add(pixptr + 3, 1);
                }
            }
        }

        for (uint i = id; i < image_size.x * image_size.y; i += n_seeds) {
            __global uint* pixptr = image + i * 4;
            float4 frgba = (float4)(pixptr[0], pixptr[1], pixptr[2], pixptr[3]);
            frgba = (fabs(frgba.a) < EPSILON) ? (float4)(0, 0, 0, 1) : (frgba / frgba.a);
            pixptr[0] = frgba.r;
            pixptr[1] = frgba.g;
            pixptr[2] = frgba.b;
            pixptr[3] = frgba.a;
        }
}