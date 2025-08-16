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

__kernel void flame_kernel(
    __global particle_t* particles,
    __global uint* histogram,
    __constant transform_t* transforms,
    __constant float4* palette,
    __constant float* camera,
    const uint n_itrs,
    const uint skip_itrs,
    const uint2 image_size,
    const uint n_transforms,
    const uint n_colors,
    const uint supersampling) {
        size_t id = get_global_id(0);
        size_t n_seeds = get_global_size(0);
        uint2 histogram_size = image_size * supersampling;

        __private particle_t particle = particles[id];

        for (uint i = 0; i < n_itrs; i++) {
            LCG32_UNIFORM(particle.seed, p);
            uint t_choice;
            for (t_choice = 0; p > 0 && t_choice < n_transforms; p -= transforms[t_choice++].probability);
            t_choice = min(t_choice, n_transforms);
            __constant transform_t* transform = transforms + t_choice - 1;

            particle = apply_transform(transform, particle);

            if (i >= skip_itrs) {
                // TODO: Final transform
                float2 pixel = affine_transform(camera, particle.xy);
                uint ux = (uint)pixel.x;
                uint uy = (uint)pixel.y;
                if (ux >= 0 && uy >= 0 && ux < histogram_size.x && uy < histogram_size.y) {
                    uint pixel_id = ux + uy * histogram_size.x;
                    __global uint* pixptr = histogram + pixel_id * 4;
                    uchar4 rgba = sample_palette(palette, particle.color, n_colors);
                    atomic_add(pixptr + 0, rgba.r);
                    atomic_add(pixptr + 1, rgba.g);
                    atomic_add(pixptr + 2, rgba.b);
                    atomic_add(pixptr + 3, 1);
                }
            }
        }

        particles[id] = particle;
}

__kernel void downsample_kernel(
    __global uint* histogram,
    __global uint* image,
    const uint2 image_size,
    const uint supersampling
    ){
        size_t id = get_global_id(0);
        size_t n_threads = get_global_size(0);
        uint2 histogram_size = image_size * supersampling;
        for (uint dest_pix = id; dest_pix < image_size.x * image_size.y; dest_pix += n_threads) {
            uint4 sum = 0;
            uint2 dest_xy = (uint2)(dest_pix % image_size.x, dest_pix / image_size.x);
            for (uint dx = 0; dx < supersampling; dx++) {
                for (uint dy = 0; dy < supersampling; dy++) {
                    uint2 src_xy = dest_xy * supersampling + (uint2)(dx, dy);
                    uint src_pix = src_xy.y * histogram_size.x + src_xy.x;
                    sum.r += histogram[src_pix * 4 + 0];
                    sum.g += histogram[src_pix * 4 + 1];
                    sum.b += histogram[src_pix * 4 + 2];
                    sum.a += histogram[src_pix * 4 + 3];
                }
            }
            //sum /= supersampling * supersampling;
            image[dest_pix * 4 + 0] = sum.r;
            image[dest_pix * 4 + 1] = sum.g;
            image[dest_pix * 4 + 2] = sum.b;
            image[dest_pix * 4 + 3] = sum.a;
        }
}

__kernel void rowmax_kernel(
    __global uint* image,
    __global uint* maxima,
    const uint2 image_size){
        size_t id = get_global_id(0);
        size_t n_threads = get_global_size(0);
        for (uint y = id; y < image_size.y; y += n_threads) {
            uint maximum = 0;
            for (uint x = 0 ; x < image_size.x; x++) {
                uint index = y * image_size.x + x;
                uint alpha = image[index * 4 + 3];
                maximum = (alpha > maximum) ? alpha : maximum;
            }
            maxima[y] = maximum;
        }
}

__kernel void tonemap_kernel(
    __global uint* image,
    const uint2 image_size,
    const float brightness,
    const float gamma,
    const float vibrancy,
    const uint max_alpha,
    const uint mode){
        size_t id = get_global_id(0);
        size_t n_threads = get_global_size(0);
        for (uint i = id; i < image_size.x * image_size.y; i += n_threads) {
            __global uint* pixptr = image + i * 4;
            float4 frgba = (float4)(pixptr[0], pixptr[1], pixptr[2], pixptr[3]);
            // Log-log scale
            if (mode & TONEMAP_MODE_LOG) {
                //frgba.a = frgba.a / log10(1 + brightness * frgba.a / max_alpha);
                frgba *= log10(1 + brightness * frgba.a / max_alpha) / frgba.a;
            } else {
                frgba = (fabs(frgba.a) < EPSILON) ? (float4)(0, 0, 0, 1) : (frgba / frgba.a);
            }
            //frgba *= pow(frgba.a, gamma) / frgba.a;
            frgba = vibrancy * frgba * pow(frgba.a, gamma) / frgba.a + (1 - vibrancy) * 256 * pow(frgba / 256, gamma);
            frgba = clamp(frgba, 0, 255);
            pixptr[0] = frgba.r;
            pixptr[1] = frgba.g;
            pixptr[2] = frgba.b;
            pixptr[3] = frgba.a;
        }
}