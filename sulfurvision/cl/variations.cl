#define VARIATION(name) float2 variation_##name(float4 xyrt, uint* seed, __constant float* params, __constant float affine[6], float weight)

VARIATION(linear) {
    return xyrt.xy;
}

VARIATION(sinusoidal) {
    return sin(xyrt.xy);
}

VARIATION(spherical) {
    return xyrt.xy / (xyrt.z * xyrt.z);
}

VARIATION(swirl) {
    float c;
    float s = sincos(xyrt.z * xyrt.z, &c);
    return (float2)(xyrt.x * s - xyrt.y * c, xyrt.x * c + xyrt.y * s);
}

VARIATION(horseshoe) {
    return (float2)(
        (xyrt.x - xyrt.y) * (xyrt.x + xyrt.y),
        2 * xyrt.x * xyrt.y
    )/ xyrt.z;
}

VARIATION(polar) {
    return (float2)(xyrt.w / M_PI_F, xyrt.z - 1);
}

VARIATION(handkerchief) {
    return xyrt.z * (float2)(sin(xyrt.w + xyrt.z), cos(xyrt.w - xyrt.z));
}

VARIATION(heart) {
    float c;
    float s = sincos(xyrt.w * xyrt.z, &c);
    return xyrt.z * (float2)(s, -c);
}

VARIATION(disc) {
    float c;
    float s = sincos(M_PI_F * xyrt.z, &c);
    return xyrt.w / M_PI_F * (float2)(s, c);
}

VARIATION(spiral) {
    float cr, ct;
    float sr = sincos(xyrt.z, &cr);
    float st = sincos(xyrt.w, &ct);
    return (float2)(ct + sr, st - cr) / xyrt.z;
}

VARIATION(hyperbolic) {
    float c;
    float s = sincos(xyrt.w, &c);
    return (float2)(s / xyrt.z, c * xyrt.z);
}

VARIATION(diamond) {
    float cr, ct;
    float sr = sincos(xyrt.z, &cr);
    float st = sincos(xyrt.w, &ct);
    return (float2)(st * cr, ct * sr);
}

VARIATION(ex) {
    float p0 = sin(xyrt.w + xyrt.z);
    p0 *= p0 * p0;
    float p1 = cos(xyrt.w - xyrt.z);
    p1 *= p1 * p1;
    return xyrt.z * (float2)(p0 + p1, p0 - p1);
}

VARIATION(julia) {
    *seed = lcg32(*seed);
    float omega = ((float)*seed / MASK32 >= 0.5) ? M_PI_F : 0.0;
    float c;
    float s = sincos(xyrt.w / 2 + omega, &c);
    return sqrt(xyrt.z) * (float2)(c, s);
}

VARIATION(bent) {
    if (xyrt.x >= 0 && xyrt.y >= 0)
        return xyrt.xy;
    if (xyrt.y >= 0)
        return (float2)(xyrt.x * 2, xyrt.y);
    if (xyrt.x >= 0)
        return (float2)(xyrt.x, xyrt.y / 2);
    return (float2)(xyrt.x * 2, xyrt.y / 2);
}

VARIATION(waves) {
    float c = (fabs(affine[2]) < EPSILON) ? 1.0 : affine[2];
    float f = (fabs(affine[5]) < EPSILON) ? 1.0 : affine[5];
    return (float2)(
        xyrt.x + affine[1] * sin(xyrt.y / (c * c)),
        xyrt.y + affine[4] * sin(xyrt.x / (f * f))
    );
}

VARIATION(fisheye) {
    return xyrt.yx * 2 / (xyrt.z + 1);
}

VARIATION(popcorn) {
    return (float2)(
        xyrt.x + affine[2] * sin(tan(3 * xyrt.y)),
        xyrt.y + affine[5] * sin(tan(3 * xyrt.x))
    );
}

VARIATION(exponential) {
    float c;
    float s = sincos(M_PI_F * xyrt.y, &c);
    return exp(xyrt.x - 1) * (float2)(c, s);
}

VARIATION(power) {
    float c;
    float s = sincos(xyrt.w, &c);
    return pow(xyrt.z, s) * (float2)(c, s);
}

VARIATION(cosine) {
    float pix = M_PI_F * xyrt.x;
    return (float2)(
        cos(pix) * cosh(xyrt.y),
        -sin(pix) * sinh(xyrt.y)
    );
}

VARIATION(rings) {
    float c;
    float s = sincos(xyrt.w, &c);
    float c2 = affine[2] * affine[2];
    return (fmod(xyrt.z + c2, 2 * c2) - c2 + xyrt.z * (1 - c2)) * (float2)(c, s);
}

VARIATION(fan) {
    float t = affine[2] * affine[2] * M_PI_F;
    float arg = (fmod(xyrt.z + affine[5], t) > t / 2)
        ? (xyrt.z - t / 2)
        : (xyrt.z + t / 2);
    float c;
    float s = sincos(arg, &c);
    return xyrt.z * (float2)(c, s);
}

VARIATION(blob) {
    float c;
    float s = sincos(xyrt.w, &c);
    return xyrt.z *
        (params[1] + (params[0] - params[1]) / 2 * (sin(params[2] * xyrt.w) + 1)) *
        (float2)(c, s);
}

VARIATION(pdj) {
    return (float2)(
        sin(params[0] * xyrt.y) - cos(params[1] * xyrt.x),
        sin(params[2] * xyrt.x) - cos(params[3] * xyrt.y)
    );
}

VARIATION(fan2) {
    float p = M_PI_F * params[0] * params[0];
    float t = xyrt.w + params[1] - p * (int)(2* xyrt.w * params[1] / p);
    float arg = (t > p / 2) ? (xyrt.w - p / 2) : (xyrt.w + p / 2);
    float c;
    float s = sincos(arg, &c);
    return xyrt.z * (float2)(s, c);
}

VARIATION(rings2) {
    float p = params[0] * params[0];
    float t = xyrt.z - 2 * p * (int)((xyrt.z + p) / (2 * p)) + xyrt.z * (1 - p);
    float c;
    float s = sincos(xyrt.w, &c);
    return t * (float2)(s, c);
}

VARIATION(eyefish) {
    return 2 / (xyrt.z + 1) * xyrt.xy;
}

VARIATION(bubble) {
    return 4 / (xyrt.z * xyrt.z + 4) * xyrt.xy;
}

VARIATION(cylinder) {
    return (float2)(sin(xyrt.x), xyrt.y);
}

VARIATION(perspective) {
    float c;
    float s = sincos(params[0], &c);
    return params[1] / (params[1] - xyrt.y * s) * (float2)(xyrt.x, xyrt.y * c);
}

VARIATION(noise) {
    LCG32_UNIFORM(*seed, phi1);
    LCG32_UNIFORM(*seed, phi2);
    float c;
    float s = sincos(2 * M_PI_F * phi2, &c);
    return phi1 * (float2)(xyrt.x * c, xyrt.y * s);
}

VARIATION(juliaN) {
    LCG32_UNIFORM(*seed, phi);
    float p3 = (int)(fabs(params[0]) * phi);
    float t = (M_PI_2_F - xyrt.w + 2 * M_PI * p3) / params[0];
    float c;
    float s = sincos(t, &c);
    return pow(xyrt.z, params[1] / params[0]) * (float2)(c, s);
}

VARIATION(juliaScope) {
    LCG32_UNIFORM(*seed, phi);
    float p3 = (int)(fabs(params[0]) * phi);
    *seed = lcg32(*seed);
    float delta = ((float)(*seed) / MASK32 > 0.5) ? 1 : -1;
    float t = (delta * (M_PI_2_F - xyrt.w) + 2 * M_PI_F * p3) / params[0];
    float c;
    float s = sincos(t, &c);
    return pow(xyrt.z, params[1] / params[0]) * (float2)(c, s);
}

VARIATION(blur) {
    LCG32_UNIFORM(*seed, phi1);
    LCG32_UNIFORM(*seed, phi2);
    float c;
    float s = sincos(M_PI_F * 2 * phi2, &c);
    return phi1 * (float2)(c, s);
}

VARIATION(gaussian) {
    float phi = -2;
    *seed = lcg32(*seed);
    phi += (float)(*seed) / MASK32;
    *seed = lcg32(*seed);
    phi += (float)(*seed) / MASK32;
    *seed = lcg32(*seed);
    phi += (float)(*seed) / MASK32;
    *seed = lcg32(*seed);
    phi += (float)(*seed) / MASK32;
    *seed = lcg32(*seed);
    float phi5 = (float)(*seed) / MASK32;
    float c;
    float s = sincos(2 * M_PI_F * phi5, &c);
    return phi * (float2)(c, s);
}

VARIATION(radialBlur) {
    float p1 = params[0] * M_PI_2_F;
    float phi = -2;
    *seed = lcg32(*seed);
    phi += (float)(*seed) / MASK32;
    *seed = lcg32(*seed);
    phi += (float)(*seed) / MASK32;
    *seed = lcg32(*seed);
    phi += (float)(*seed) / MASK32;
    *seed = lcg32(*seed);
    phi += (float)(*seed) / MASK32;
    phi *= weight;
    float cp1;
    float sp1 = sincos(p1, &cp1);
    float t2 = M_PI_2_F - xyrt.w + phi * sp1;
    float t3 = phi * cp1 - 1;
    float ct2;
    float st2 = sincos(t2, &ct2);
    return (float2)(xyrt.z * ct2 + t3 * xyrt.x, xyrt.z * st2 + t3 * xyrt.y) / (fabs(weight) > EPSILON ? weight : 1);
}

VARIATION(pie) {
    LCG32_UNIFORM(*seed, phi1);
    LCG32_UNIFORM(*seed, phi2);
    LCG32_UNIFORM(*seed, phi3);
    float t1 = (int)(phi1 * params[0] + .5);
    float t2 = params[1] + M_PI_2_F / params[0] * (t1 + phi2 * params[2]);
    float c;
    float s = sincos(t2, &c);
    return phi3 * (float2)(c, s);
}

VARIATION(ngon) {
    float p2 = 2 * M_PI_F / params[1];
    float psi = M_PI_2_F - xyrt.z;
    float t3 = psi - p2 * floor(psi / p2);
    float t4 = (t3 > p2 / 2) ? t3 : (t3 - p2);
    float k = (params[2] * (1 / cos(t4) - 1) + params[3]) / pow(xyrt.z, params[0]);
    return k * xyrt.xy;
}

VARIATION(curl) {
    float t1 = 1 + params[0] * xyrt.x + params[1] * (xyrt.x * xyrt.x + xyrt.y * xyrt.y);
    float t2 = params[0] * xyrt.y + 2 * params[1] * xyrt.x * xyrt.y;
    return (float2)(xyrt.x * t1 + xyrt.y * t2, xyrt.y * t1 - xyrt.x * t2) / (t1 * t1 + t2 * t2);
}

VARIATION(rectangles) {
    return (float2)(
        2 * (floor(xyrt.x / params[0]) + 1) * params[0] - xyrt.x,
        2 * (floor(xyrt.y / params[1]) + 1) * params[1] - xyrt.y
    );
}

VARIATION(arch) {
    LCG32_UNIFORM(*seed, phi);
    float c;
    float s = sincos(phi * M_PI_F * weight, &c);
    return (float2)(s, s * s / c);
}

VARIATION(tangent) {
    return (float2)(sin(xyrt.x) / cos(xyrt.y), tan(xyrt.y));
}

VARIATION(square) {
    LCG32_UNIFORM(*seed, phi1);
    LCG32_UNIFORM(*seed, phi2);
    return (float2)(phi1 - 0.5, phi2 - 0.5);
}

VARIATION(rays) {
    LCG32_UNIFORM(*seed, phi);
    return weight * tan(phi * M_PI_F * weight) / (xyrt.z * xyrt.z) * (float2)(cos(xyrt.x), sin(xyrt.y));
}

VARIATION(blade) {
    LCG32_UNIFORM(*seed, phi);
    float c;
    float s = sincos(phi * xyrt.z * weight, &c);
    return xyrt.x * (float2)(c + s, c - s);
}

VARIATION(secant) {
    weight = (fabs(weight) > EPSILON) ? weight : 1;
    return (float2)(xyrt.x, 1 / (weight * cos(weight * xyrt.z)));
}

VARIATION(twintrain) {
    LCG32_UNIFORM(*seed, phi);
    float c;
    float s = sincos(phi * xyrt.z * weight, &c);
    float t = log10(s * s) + c;
    return xyrt.x * (float2)(t, t - M_PI_F * s);
}

VARIATION(cross) {
    float z = xyrt.x * xyrt.x - xyrt.y * xyrt.y;
    return sqrt(1 / (z * z)) * xyrt.xy;
}

#undef VARIATION
