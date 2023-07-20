#pragma once

#include "camera.cuh"
#include "curand_kernel.h"
#include "hittable.cuh"
#include "material.cuh"
#include "ray.cuh"

__global__ void g_renderInit(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__device__ Color rayColor(const Ray &r, Hittable **world,
                          curandState *local_rand_state, int max_depth) {
    Ray cur_ray = r;
    Color cur_attenuation = Color(1.0, 1.0, 1.0);
    for (int i = 0; i < max_depth; i++) {
        HitRecord rec;
        if ((*world)->hit(cur_ray, 0.001f, INFINITY, rec)) {
            Ray scattered;
            Color attenuation;
            if (rec.material->scatter(local_rand_state, cur_ray, rec,
                                      attenuation, scattered)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return Color(0.0, 0.0, 0.0);
            }
        } else {
            Vec3 unit_direction = unitVector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            Color c = (1.0f - t) * Color(1.0f, 1.0f, 1.0f) +
                      t * Color(0.5f, 0.7f, 1.0f);
            return cur_attenuation * c;
        }
    }
    return Color(0.0, 0.0, 0.0);
}

// use recursion
Color rayColor(const Ray &r, Hittable **world, int depth) {
    if (depth <= 0) {
        return Color(0.0, 0.0, 0.0);
    }
    HitRecord rec;
    if ((*world)->hit(r, 0.001f, INFINITY, rec)) {
        Ray scattered;
        Color attenuation;
        if (rec.material->scatter(r, rec, attenuation, scattered)) {
            return attenuation * rayColor(scattered, world, depth - 1);
        }
        return Color(0.0, 0.0, 0.0);
    }

    Vec3 unit_direction = unitVector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    Color c =
        (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);
    return c;
}


__global__ void g_render(int max_x, int max_y, int ns, int max_depth,
                         Camera **cam, Hittable **world,
                         curandState *rand_state, Color *fb) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    Color col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + randf(&local_rand_state)) / float(max_x);
        float v = float(j + randf(&local_rand_state)) / float(max_y);
        Ray r = (*cam)->getRay(&local_rand_state, u, v);
        col += rayColor(r, world, &local_rand_state, max_depth);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col = Color(sqrt(col.x()), sqrt(col.y()), sqrt(col.z()));
    fb[pixel_index] = col;
}

void render(int max_x, int max_y, int ns, int max_depth, Camera **cam,
            Hittable **world, Color *fb) {
    for (int j = max_y - 1; j >= 0; j--) {
        for (int i = 0; i < max_x; i++) {
            Color col(0, 0, 0);
            for (int s = 0; s < ns; s++) {
                float u = float(i + randf()) / float(max_x);
                float v = float(j + randf()) / float(max_y);
                Ray r = (*cam)->getRay(u, v);
                col += rayColor(r, world, max_depth);
            }
            col /= float(ns);
            col = Color(sqrt(col.x()), sqrt(col.y()), sqrt(col.z()));
            fb[j * max_x + i] = col;
        }
    }
}