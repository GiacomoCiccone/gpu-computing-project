#include <iostream>
#include <fstream>

#include "render_option.h"
#include "camera.cuh"
#include "cuda_helpers.cuh"
#include "scene.cuh"
#include "render.cuh"



int main(int argc, char** argv) {
    // warm up GPU
    cudaFree(0);

    RenderOption opt = RenderOption::parseCommandLine(argc, argv);
    int nx = opt.width;
    int ny = opt.height;
    int ns = opt.num_samples;
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(Color);

    // allocate FB
    Color* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    if (opt.use_gpu) {
        int tx = 8;
        int ty = 8;

        // allocate random state for creating world
        curandState* d_rand_state_world;
        checkCudaErrors(cudaMalloc((void**)&d_rand_state_world, sizeof(curandState)));

        // create world
        Hittable** d_list;
        //checkCudaErrors(cudaMalloc((void**)&d_list, 5 * sizeof(Hittable*)));
        checkCudaErrors(cudaMalloc((void**)&d_list, (22 * 22 + 1 + 3) * sizeof(Hittable*)));
        Hittable** d_world;
        checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Hittable*)));
        Camera** d_camera;
        checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));
        g_createWorld<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state_world);
        checkCudaErrors(cudaDeviceSynchronize());

        // allocate random state for rendering
        curandState* d_rand_state;
        checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
        
        // render our buffer
        dim3 blocks(nx / tx + 1, ny / ty + 1);
        dim3 threads(tx, ty);
        
        double start_time = clock();
        g_renderInit<<<blocks, threads>>>(nx, ny, d_rand_state);
        checkCudaErrors(cudaDeviceSynchronize());

        g_render<<<blocks, threads>>>(nx, ny, ns, opt.max_depth, d_camera, d_world, d_rand_state, fb);
        checkCudaErrors(cudaDeviceSynchronize());
        double end_time = clock();
        std::cout << "Render time on GPU: " << (end_time - start_time) / CLOCKS_PER_SEC << "s\n";

        // clean up
        g_freeWorld<<<1, 1>>>(d_list, d_world, d_camera);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaFree(d_rand_state));
        checkCudaErrors(cudaFree(d_rand_state_world));
        checkCudaErrors(cudaFree(d_list));
        checkCudaErrors(cudaFree(d_world));
        checkCudaErrors(cudaFree(d_camera));
    } else {
        std::cout << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel on CPU\n";

        // create world
        Hittable** list = new Hittable*[22*22+1+3];
        Hittable** world = new Hittable*[1];
        Camera** camera = new Camera*[1];
        createWorld(list, world, camera, nx, ny);

        // render our buffer
        double start_time = clock();
        render(nx, ny, ns, opt.max_depth, camera, world, fb);
        double end_time = clock();
        std::cout << "Render time on CPU: " << (end_time - start_time) / CLOCKS_PER_SEC << "s\n";

        // clean up
        freeWorld(list, world, camera);
        delete[] list;
        delete[] world;
        delete[] camera;
    }

    // Output FB as Image
    std::ofstream ofs;
    ofs.open(opt.output_file);
    ofs << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * fb[pixel_index].x());
            int ig = int(255.99 * fb[pixel_index].y());
            int ib = int(255.99 * fb[pixel_index].z());
            ofs << ir << " " << ig << " " << ib << "\n";
        }
    }
    std::cout << "Image written to " << opt.output_file << std::endl;

    // clean up    
    ofs.close();
    checkCudaErrors(cudaFree(fb));
    cudaDeviceReset();
    return 0;
}
