particle_t apply_transform(
    __constant transform_t* transform,
    const particle_t particle
) {
    // Lerp color
    float color = particle.color;
    color += (transform->color - color) * transform->color_speed;
    float2 xy = affine_transform(transform->affine, particle.xy);
    uint seed = particle.seed;

    // Precalculate extra params
    float r = hypot(xy.x, xy.y);
    float theta = atan2(xy.y, xy.x);
    float4 xyrt = (float4)(xy, r, theta);
    float2 new_xy = 0;

@@VARIATIONS@@

    return (particle_t){new_xy, seed, color};
}

// TODO: take in particle states, separate out into functions
__kernel void flame_kernel(
    __global particle_t* particles,
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

        __private particle_t particle = particles[id];

        for (uint i = 0; i < n_itrs; i++) {
            LCG32_UNIFORM(particle.seed, p);
            uint t_choice;
            for (t_choice = 0; p >= 0; p -= transforms[t_choice++].probability);
            __constant transform_t* transform = transforms + t_choice - 1;

            particle = apply_transform(transform, particle);

            if (i >= skip_itrs) {
                // TODO: Final transform
                float2 pixel = affine_transform(camera, particle.xy);
                uint ux = (uint)pixel.x;
                uint uy = (uint)pixel.y;
                if (ux >= 0 && uy >= 0 && ux < image_size.x && uy < image_size.y) {
                    uint pixel_id = ux + uy * image_size.x;
                    __global uint* pixptr = image + pixel_id * 4;
                    uchar4 rgba = sample_palette(palette, particle.color, n_colors);
                    atomic_add(pixptr + 0, rgba.r);
                    atomic_add(pixptr + 1, rgba.g);
                    atomic_add(pixptr + 2, rgba.b);
                    atomic_add(pixptr + 3, 1);
                }
            }
        }

        particles[id] = particle;

        // TODO log-density, gamma, and vibrancy coloration, supersampling
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