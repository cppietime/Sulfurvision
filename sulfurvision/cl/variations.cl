#define VARIATION(name) float2 variation_##name(float4 xyrt, __global uint* seed, __constant float* params, __constant float* affine)

VARIATION(linear) {
    return xyrt.xy;
}
