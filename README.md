#sulfurvision#

I'm making this little project to try to generate some nice fractals in Python, hopefully using the GPU effectively.

Steps remaining:

- (De)Serialization
- Interpolation between Transforms
- Python-side data structures
- OpenCL:

  - Detect and choose device
  - Discover available memory and capabilities
  - Device-side structures
  - Port each variation function
  - Write actual kernel

OpenCL kernels IO:

- Meta parameters:

  - Number of transforms (could be primitive)
- Constant inputs:

  - All transform fields (weights, params, affine, probability, color, color_speed) x NUM_TRANSFORMS
  - Flame fields: color palette, camera affine
- Primitive inputs:

  - Image dimensions
  - Number of iterations
  - Skipped iterations
  - Seed multiplication factor?
- Global inputs/outputs:

  - Seeds
  - Image grid
  - New seeds output?
