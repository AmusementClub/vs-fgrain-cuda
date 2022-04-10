# vs-fgrain-cuda
[Realistic Film Grain Rendering](https://www.ipol.im/pub/art/2017/192/) for VapourSynth, implemented in CUDA

## Usage
Prototype:

`core.fgrain_cuda.Add(clip clip[, int num_iterations = 800, float grain_radius_mean = 0.1, float grain_radius_std=0.0, float sigma = 0.8, int seed = 0])`

Currently only grays is supported.

## Compilation
[CUDA](https://developer.nvidia.com/cuda-downloads) is required.

```bash
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release \
-D CMAKE_CUDA_ARCHITECTURES="52;61-real;75-real;86-real" \
-D CMAKE_CUDA_FLAGS="--threads 0 --use_fast_math"

cmake --build build

cmake --install build
```

