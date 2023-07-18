#include <iostream>

#include "render_option.h"
#include "ray.cuh"
#include "random.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

__global__ void dummyKernel(curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1984, i, 0, &rand_state[i]);

    printf("Random number GPU: %f\n", randf(&rand_state[i]));

    printf("Random vec3 GPU: %f %f %f\n", randVec3(&rand_state[i]).x(), randVec3(&rand_state[i]).y(), randVec3(&rand_state[i]).z());

}

int main(int argc, char** argv) {

    std::cout << "Random number CPU: " << randf() << "\n";

    std::cout << "Random vec3 CPU: " << randVec3() << "\n";

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, sizeof(curandState)));

    dummyKernel<<<1, 1>>>(d_rand_state);
    checkCudaErrors(cudaDeviceSynchronize());


    return 0;
}
