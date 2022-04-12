#include <cuda_runtime.h>

void film_grain_rendering(
    float * __restrict__ d_dst,
    const float * __restrict__ d_src,
    int width,
    int height,
    int stride,
    int num_iterations,
    float grain_radius_mean,
    float grain_radius_std,
    float sigma,
    int seed,
    const float * __restrict__ d_lambda,
    const float * __restrict__ d_exp_lambda,
    const float * __restrict__ d_x_gaussian,
    const float * __restrict__ d_y_gaussian,
    cudaStream_t stream
) noexcept;

__device__
static unsigned int wang_hash(unsigned int seed) noexcept {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 668265261u;
    seed = seed ^ (seed >> 15u);
    return seed;
}

__device__
static unsigned int cellseed(unsigned int x, unsigned int y, unsigned int offset) noexcept {
    const unsigned int period = 65536u; // 65536 = 2^16
    unsigned int s = ((y % period) * period + (x % period)) + offset;
    if (s == 0u) s = 1u;
    return s;
}

struct noise_prng {
    unsigned int state;
    __device__ noise_prng() = delete;
    __device__ noise_prng(unsigned int seed) noexcept : state{ wang_hash(seed) } {}
};

__device__
static unsigned int myrand(noise_prng & p) noexcept {
    p.state ^= p.state << 13u;
    p.state ^= p.state >> 17u;
    p.state ^= p.state << 5u;
    return p.state;
}

__device__
static float myrand_uniform_0_1(noise_prng & p) noexcept {
    return (float) myrand(p) / (float) 4294967295u;
}

__device__
static int my_rand_poisson(noise_prng & p, float lambda, float prod) noexcept {
    /* Inverse transform sampling */
    float u = myrand_uniform_0_1(p);

    float sum = prod;
    float x {};
    while ((u > sum) && (x < floorf(10000.0f * lambda))) {
        x += 1.0f;
        prod *= lambda / x;
        sum += prod;
    }

    return (int) x;
}

__device__
static float sq_distance(float x1, float y1, float x2, float y2) noexcept {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

__device__
static float render_pixel(
    const float * __restrict__ src,
    int width,
    int height,
    int stride,
    int x,
    int y,
    int num_iterations,
    float grain_radius_mean,
    float grain_radius_std,
    float sigma,
    int seed,
    const float * __restrict__ lambda,
    const float * __restrict__ exp_lambda,
    const float * __restrict__ x_gaussian,
    const float * __restrict__ y_gaussian
) noexcept {

    const float inv_grain_radius_mean = ceilf(1.0f / grain_radius_mean);
    const float ag = 1.0f / inv_grain_radius_mean;

    // assert (grain_radius_std == 0);

    // auto p = noise_prng{ cellseed(x, y, seed) };

    int pixel_val {};
    for (int i = 0; i < num_iterations; i++) {
        float x_gauss = x + sigma * x_gaussian[i];
        float y_gauss = y + sigma * y_gaussian[i];

        int x_start = __float2int_rz((x_gauss - grain_radius_mean) * inv_grain_radius_mean);
        int x_end = __float2int_ru((x_gauss + grain_radius_mean) * inv_grain_radius_mean);
        int y_start = __float2int_rz((y_gauss - grain_radius_mean) * inv_grain_radius_mean);
        int y_end = __float2int_ru((y_gauss + grain_radius_mean) * inv_grain_radius_mean);

        for (int ix = x_start; ix <= x_end; ix++) {
            for (int iy = y_start; iy <= y_end; iy++) {
                float2 cell_corner { ag * ix, ag * iy };
                auto p = noise_prng{ cellseed(ix, iy, seed) };

                int px = fminf(fmaxf(/*floorf*/roundf(cell_corner.x), 0.0f), width - 1);
                int py = fminf(fmaxf(/*floorf*/roundf(cell_corner.y), 0.0f), height - 1);
                int index = max(0, min(__float2int_rz(src[py * stride + px] * 255.1f), 255));

                int n_cell = my_rand_poisson(p, lambda[index], exp_lambda[index]);

                for (int k = 0; k < n_cell; k++) {
                    float xCentreGrain = cell_corner.x + ag * myrand_uniform_0_1(p);
                    float yCentreGrain = cell_corner.y + ag * myrand_uniform_0_1(p);
                    if (sq_distance(xCentreGrain, yCentreGrain, x_gauss, y_gauss) < grain_radius_mean * grain_radius_mean) {
                        pixel_val += 1;
                        goto NEXT_MC;
                    }
                }
            }
        }
NEXT_MC:
        ;
    }

    return pixel_val / (float) num_iterations;
}

__global__ __launch_bounds__(128)
static void film_grain_rendering_kernel(
    float * __restrict__ dst,
    const float * __restrict__ src,
    int width,
    int height,
    int stride,
    int num_iterations,
    float grain_radius_mean,
    float grain_radius_std,
    float sigma,
    int seed,
    const float * __restrict__ lambda,
    const float * __restrict__ exp_lambda,
    const float * __restrict__ x_gaussian,
    const float * __restrict__ y_gaussian
) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return ;
    }

    dst[y * stride + x] = render_pixel(
        src, width, height, stride,
        x, y, num_iterations, grain_radius_mean, grain_radius_std, sigma, seed, lambda, exp_lambda, x_gaussian, y_gaussian
    );
}

void film_grain_rendering(
    float * __restrict__ d_dst,
    const float * __restrict__ d_src,
    int width,
    int height,
    int stride,
    int num_iterations,
    float grain_radius_mean,
    float grain_radius_std,
    float sigma,
    int seed,
    const float * __restrict__ d_lambda,
    const float * __restrict__ d_exp_lambda,
    const float * __restrict__ d_x_gaussian,
    const float * __restrict__ d_y_gaussian,
    cudaStream_t stream
) noexcept {

    const dim3 block { 32, 4 };

    const dim3 grid {
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    };

    film_grain_rendering_kernel<<<grid, block, 0, stream>>>(d_dst, d_src, width, height, stride, num_iterations, grain_radius_mean, grain_radius_std, sigma, seed, d_lambda, d_exp_lambda, d_x_gaussian, d_y_gaussian);
}
