#pragma once

#include "camera.cuh"
#include "hittable_list.cuh"
#include "material.cuh"
#include "sphere.cuh"

// __global__ void g_createWorld(Hittable **d_list, Hittable **d_world,
//                               Camera **d_camera, int width, int height,
//                               curandState *rand_state) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         *(d_list) = new Sphere(Point3(0, 0, -1), 0.5,
//                                new Lambertian(Color(0.1, 0.2, 0.5)));
//         *(d_list + 1) = new Sphere(Point3(0, -100.5, -1), 100,
//                                    new Lambertian(Color(0.8, 0.8, 0.0)));
//         *(d_list + 2) = new Sphere(Point3(1, 0, -1), 0.5,
//                                    new Metal(Color(0.8, 0.6, 0.2), 0.3));
//         *(d_list + 3) = new Sphere(Point3(-1, 0, -1), 0.5, new
//         Dielectric(1.5));
//         *(d_list + 4) =
//             new Sphere(Point3(-1, 0, -1), -0.45, new Dielectric(1.5));
//         *d_world = new HittableList(d_list, 5);

//         Point3 lookfrom(3, 3, 2);
//         Point3 lookat(0, 0, -1);
//         float dist_to_focus = (lookfrom - lookat).length();
//         float aperture = 2.0;
//         *d_camera =
//             new Camera(lookfrom, lookat, Vec3(0, 1, 0), 20,
//                        float(width) / float(height), aperture,
//                        dist_to_focus);
//     }
// }

#define RND (randf(rand_state))

__global__ void g_createWorld(Hittable **d_list, Hittable **d_world,
                              Camera **d_camera, int width, int height,
                              curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1234, 0, 0, &rand_state[0]);
        d_list[0] = new Sphere(Point3(0, -1000.0, -1), 1000,
                               new Lambertian(Color(0.5, 0.5, 0.5)));
        int i = 1;

        #pragma unroll
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                Vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) {
                    d_list[i++] = new Sphere(
                        center, 0.2,
                        new Lambertian(Color(RND * RND, RND * RND, RND * RND)));
                } else if (choose_mat < 0.95f) {
                    d_list[i++] =
                        new Sphere(center, 0.2,
                                   new Metal(Color(0.5f * (1.0f + RND),
                                                   0.5f * (1.0f + RND),
                                                   0.5f * (1.0f + RND)),
                                             0.5f * RND));
                } else {
                    d_list[i++] = new Sphere(center, 0.2, new Dielectric(1.5));
                }
            }
        }
        d_list[i++] = new Sphere(Point3(0, 1, 0), 1.0, new Dielectric(1.5));
        d_list[i++] = new Sphere(Point3(-4, 1, 0), 1.0,
                                 new Lambertian(Color(0.4, 0.2, 0.1)));
        d_list[i++] = new Sphere(Point3(4, 1, 0), 1.0,
                                 new Metal(Color(0.7, 0.6, 0.5), 0.0));
        *d_world = new HittableList(d_list, 22 * 22 + 1 + 3);

        Point3 lookfrom(13, 2, 3);
        Point3 lookat(0, 0, 0);
        float dist_to_focus = 10.0;
        (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera =
            new Camera(lookfrom, lookat, Vec3(0, 1, 0), 30.0,
                       float(width) / float(height), aperture, dist_to_focus);
    }
}

// void createWorld(Hittable **list, Hittable **world, Camera **camera, int
// widht,
//                  int weight) {
//     *(list) =
//         new Sphere(Point3(0, 0, -1), 0.5, new Lambertian(Color(0.1, 0.2,
//         0.5)));
//     *(list + 1) = new Sphere(Point3(0, -100.5, -1), 100,
//                              new Lambertian(Color(0.8, 0.8, 0.0)));
//     *(list + 2) =
//         new Sphere(Point3(1, 0, -1), 0.5, new Metal(Color(0.8, 0.6, 0.2),
//         0.3));
//     *(list + 3) = new Sphere(Point3(-1, 0, -1), 0.5, new Dielectric(1.5));
//     *(list + 4) = new Sphere(Point3(-1, 0, -1), -0.45, new Dielectric(1.5));
//     *world = new HittableList(list, 5);

//     Point3 lookfrom(3, 3, 2);
//     Point3 lookat(0, 0, -1);
//     float dist_to_focus = (lookfrom - lookat).length();
//     float aperture = 2.0;
//     *camera = new Camera(lookfrom, lookat, Vec3(0, 1, 0), 20,
//                          float(widht) / float(weight), aperture,
//                          dist_to_focus);
// }

void createWorld(Hittable **list, Hittable **world, Camera **camera, int widht,
                 int weight) {
    *(list) = new Sphere(Point3(0, -1000.0, -1), 1000,
                         new Lambertian(Color(0.5, 0.5, 0.5)));
    int i = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = randf();
            Vec3 center(a + randf(), 0.2, b + randf());
            if (choose_mat < 0.8f) {
                list[i++] = new Sphere(
                    center, 0.2,
                    new Lambertian(Color(randf() * randf(), randf() * randf(),
                                         randf() * randf())));
            } else if (choose_mat < 0.95f) {
                list[i++] = new Sphere(center, 0.2,
                                       new Metal(Color(0.5f * (1.0f + randf()),
                                                       0.5f * (1.0f + randf()),
                                                       0.5f * (1.0f + randf())),
                                                 0.5f * randf()));
            } else {
                list[i++] = new Sphere(center, 0.2, new Dielectric(1.5));
            }
        }
    }
    list[i++] = new Sphere(Point3(0, 1, 0), 1.0, new Dielectric(1.5));
    list[i++] =
        new Sphere(Point3(-4, 1, 0), 1.0, new Lambertian(Color(0.4, 0.2, 0.1)));
    list[i++] =
        new Sphere(Point3(4, 1, 0), 1.0, new Metal(Color(0.7, 0.6, 0.5), 0.0));
    *world = new HittableList(list, 22 * 22 + 1 + 3);

    Point3 lookfrom(13, 2, 3);
    Point3 lookat(0, 0, 0);
    float dist_to_focus = 10.0;
    (lookfrom - lookat).length();
    float aperture = 0.1;
    *camera = new Camera(lookfrom, lookat, Vec3(0, 1, 0), 30.0,
                         float(widht) / float(weight), aperture, dist_to_focus);
}

__global__ void g_freeWorld(Hittable **d_list, Hittable **d_world,
                            Camera **d_camera) {
    // for (int i = 0; i < 5; i++) {

    #pragma unroll
    for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
        delete ((Sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

void freeWorld(Hittable **list, Hittable **world, Camera **camera) {
    // for (int i = 0; i < 5; i++) {
    for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
        delete ((Sphere *)list[i])->mat_ptr;
        delete list[i];
    }
    delete *world;
    delete *camera;
}