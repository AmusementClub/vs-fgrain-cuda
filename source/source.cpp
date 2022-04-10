#include <cmath>
#include <memory>
#include <numbers>
#include <random>
#include <string>

#include <VapourSynth.h>
#include <VSHelper.h>

#include <cuda_runtime.h>

namespace {
    using namespace std::string_literals;
}

#define checkError(expr) do {                                            \
    if (cudaError_t result = expr; result != cudaSuccess) [[unlikely]] { \
        const char * error_str = cudaGetErrorString(result);             \
        return set_error("'"s + # expr + "' failed: " + error_str);      \
    }                                                                    \
} while(0)

extern void film_grain_rendering(
    float * d_dst,
    const float * d_src,
    int width,
    int height,
    int stride,
    int num_iterations,
    float grain_radius_mean,
    float grain_radius_std,
    float sigma,
    int seed,
    const float * d_lambda,
    const float * d_exp_lambda,
    const float * d_x_gaussian,
    const float * d_y_gaussian,
    cudaStream_t stream
);

struct FGrainData {
    VSNodeRef * node;
    int seed;
    int num_iterations;
    float grain_radius_mean;
    float grain_radius_std;
    float sigma;

    cudaStream_t stream;
    float * h_src;
    float * d_src;
    float * d_dst;
    float * h_dst;
    int d_pitch;
    float * d_lambda;
    float * d_exp_lambda;
    float * d_x_gaussian;
    float * d_y_gaussian;
};

static void VS_CC fgrainInit(
    VSMap * in,
    VSMap * out,
    void ** instanceData,
    VSNode * node,
    VSCore * core,
    const VSAPI * vsapi
) {

    const auto * d = reinterpret_cast<const FGrainData *>(*instanceData);
    vsapi->setVideoInfo(vsapi->getVideoInfo(d->node), 1, node);
}

static const VSFrameRef * VS_CC fgrainGetFrame(
    int n,
    int activationReason,
    void ** instanceData,
    void ** frameData,
    VSFrameContext * frameCtx,
    VSCore * core,
    const VSAPI * vsapi
) {

    auto * d = reinterpret_cast<FGrainData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef * src_frame = vsapi->getFrameFilter(n, d->node, frameCtx);

        VSFrameRef * dst_frame {};

        auto vi = vsapi->getVideoInfo(d->node);

        auto set_error = [&](const std::string & error_message) -> const VSFrameRef * {
            if (dst_frame != nullptr) {
                vsapi->freeFrame(src_frame);
            }
            vsapi->freeFrame(src_frame);
            vsapi->setFilterError(error_message.c_str(), frameCtx);
            return nullptr;
        };

        const auto * srcp = vsapi->getReadPtr(src_frame, 0);
        auto src_pitch = vsapi->getStride(src_frame, 0);

        vs_bitblt(d->h_src, d->d_pitch, srcp, src_pitch, vi->width * sizeof(float), vi->height);

        checkError(cudaMemcpy2DAsync(
            d->d_src, d->d_pitch,
            d->h_src, d->d_pitch,
            vi->width * sizeof(float), vi->height,
            cudaMemcpyHostToDevice, d->stream));

        film_grain_rendering(
            d->d_dst, d->d_src,
            vi->width, vi->height, d->d_pitch / sizeof(float),
            d->num_iterations, d->grain_radius_mean, d->grain_radius_std, d->sigma,
            d->seed, d->d_lambda, d->d_exp_lambda, d->d_x_gaussian, d->d_y_gaussian, d->stream);

        checkError(cudaMemcpy2DAsync(
            d->h_dst, d->d_pitch,
            d->d_dst, d->d_pitch,
            vi->width * sizeof(float), vi->height,
            cudaMemcpyDeviceToHost, d->stream));

        dst_frame = vsapi->newVideoFrame(vi->format, vi->width, vi->height, src_frame, core);
        auto * dstp = vsapi->getWritePtr(dst_frame, 0);
        auto dst_pitch = vsapi->getStride(dst_frame, 0);

        checkError(cudaStreamSynchronize(d->stream));

        vsapi->freeFrame(src_frame);

        vs_bitblt(dstp, dst_pitch, d->h_dst, d->d_pitch, vi->width * sizeof(float), vi->height);

        return dst_frame;
    }

    return nullptr;
}

static void VS_CC fgrainFree(
    void * instanceData,
    VSCore * core,
    const VSAPI * vsapi
) {
    auto * d = reinterpret_cast<FGrainData *>(instanceData);

    cudaFree(d->d_y_gaussian);
    cudaFree(d->d_x_gaussian);
    cudaFree(d->d_exp_lambda);
    cudaFree(d->d_lambda);
    cudaFreeHost(d->h_dst);
    cudaFree(d->d_dst);
    cudaFree(d->d_src);
    cudaFreeHost(d->h_src);
    cudaStreamDestroy(d->stream);

    vsapi->freeNode(d->node);

    delete d;
}

static void VS_CC fgrainCreate(
    const VSMap * in,
    VSMap * out,
    void * userData,
    VSCore * core,
    const VSAPI * vsapi
) {

    auto d = std::make_unique<FGrainData>();

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);

    auto set_error = [&](const std::string & error_message) -> void {
        vsapi->freeNode(d->node);
        vsapi->setError(out, error_message.c_str());
    };

    auto vi = vsapi->getVideoInfo(d->node);
    if (vi->format->id != pfGrayS) {
        return set_error("requires grays format");
    }

    int err;

    d->num_iterations = int64ToIntS(vsapi->propGetInt(in, "num_iterations", 0, &err));
    if (err) {
        d->num_iterations = 800;
    }

    d->grain_radius_mean = (float) vsapi->propGetFloat(in, "grain_radius_mean", 0, &err);
    if (err) {
        d->grain_radius_mean = 0.1f;
    }

    d->grain_radius_std = (float) vsapi->propGetFloat(in, "grain_radius_std", 0, &err);
    if (err) {
        d->grain_radius_std = 0.0f;
    }

    d->sigma = (float) vsapi->propGetFloat(in, "sigma", 0, &err);
    if (err) {
        d->sigma = 0.8f;
    }

    d->seed = int64ToIntS(vsapi->propGetInt(in, "seed", 0, &err));
    if (err) {
        d->seed = 0;
    }

    checkError(cudaStreamCreateWithFlags(&d->stream, cudaStreamNonBlocking));
    d->d_pitch = vi->width * sizeof(float);
    checkError(cudaMallocHost(&d->h_src, vi->height * d->d_pitch));
    checkError(cudaMalloc(&d->d_src, vi->height * d->d_pitch));
    checkError(cudaMalloc(&d->d_dst, vi->height * d->d_pitch));
    checkError(cudaMallocHost(&d->h_dst, vi->height * d->d_pitch));

    checkError(cudaMalloc(&d->d_lambda, 256 * sizeof(float)));
    checkError(cudaMalloc(&d->d_exp_lambda, 256 * sizeof(float)));
    {
        float * h_lambda;
        float * h_exp_lambda;
        checkError(cudaMallocHost(&h_lambda, 256 * sizeof(float)));
        checkError(cudaMallocHost(&h_exp_lambda, 256 * sizeof(float)));

        const float ag = 1.0 / ceil(1.0 / d->grain_radius_mean);
        for (int i = 0; i < 256; i++) {
            h_lambda[i] = -((ag * ag) / (
                std::numbers::pi_v<float> *
                (d->grain_radius_mean * d->grain_radius_mean + d->grain_radius_std * d->grain_radius_std)
            )) * std::log((255 - i) / 255.1f);
            h_exp_lambda[i] = std::exp(-h_lambda[i]);
        }

        checkError(cudaMemcpyAsync(
            d->d_lambda, h_lambda, 256 * sizeof(float),
            cudaMemcpyHostToDevice, d->stream));
        checkError(cudaMemcpyAsync(
            d->d_exp_lambda, h_exp_lambda, 256 * sizeof(float),
            cudaMemcpyHostToDevice, d->stream));

        checkError(cudaStreamSynchronize(d->stream));

        checkError(cudaFreeHost(h_exp_lambda));
        checkError(cudaFreeHost(h_lambda));
    }

    checkError(cudaMalloc(&d->d_x_gaussian, d->num_iterations * sizeof(float)));
    checkError(cudaMalloc(&d->d_y_gaussian, d->num_iterations * sizeof(float)));
    {
        std::mt19937 rng(d->seed);
        std::normal_distribution<float> normalDistribution(0.0f, d->sigma);

        float * h_x_gaussian;
        float * h_y_gaussian;
        checkError(cudaMallocHost(&h_x_gaussian, d->num_iterations * sizeof(float)));
        checkError(cudaMallocHost(&h_y_gaussian, d->num_iterations * sizeof(float)));

        for (int i = 0; i < d->num_iterations; i++) {
            h_x_gaussian[i] = normalDistribution(rng);
            h_y_gaussian[i] = normalDistribution(rng);
        }

        checkError(cudaMemcpyAsync(
            d->d_x_gaussian, h_x_gaussian, d->num_iterations * sizeof(float),
            cudaMemcpyHostToDevice, d->stream));
        checkError(cudaMemcpyAsync(
            d->d_y_gaussian, h_y_gaussian, d->num_iterations * sizeof(float),
            cudaMemcpyHostToDevice, d->stream));

        checkError(cudaStreamSynchronize(d->stream));

        checkError(cudaFreeHost(h_y_gaussian));
        checkError(cudaFreeHost(h_x_gaussian));
    }

    vsapi->createFilter(
        in, out,
        "FGrain", fgrainInit, fgrainGetFrame, fgrainFree,
        fmParallelRequests, nfNoCache, d.release(), core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin configFunc,
    VSRegisterFunction registerFunc,
    VSPlugin * plugin
) {

    configFunc(
        "io.github.amusementclub.vs-fgrain-cuda",
        "fgrain_cuda", "Realistic Film Grain Generator", VAPOURSYNTH_API_VERSION, 1, plugin);

    registerFunc("Add",
        "clip:clip;"
        "num_iterations:int:opt;"
        "grain_radius_mean:float:opt;"
        "grain_radius_std:float:opt;"
        "sigma:float:opt;"
        "seed:int:opt;",
        fgrainCreate, nullptr, plugin
    );
}
