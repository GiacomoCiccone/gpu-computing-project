#include <iostream>

#include "render_option.h"
#include "ray.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}


int main(int argc, char** argv) {
    RenderOption options = RenderOption::parseCommandLine(argc, argv);

    std::cout << "Options: " << options << std::endl;

    Ray ray(Vec3(0, 0, 0), unitVector(Vec3(1, 1, 1)));
    std::cout << ray << std::endl;
    Ray reflected_ray = reflect(ray, Vec3(0, 1, 0));
    std::cout << reflected_ray << std::endl;
    Ray refracted_ray;
    refract(ray, Vec3(0, 1, 0), .04, refracted_ray);
    std::cout << refracted_ray << std::endl;


}
