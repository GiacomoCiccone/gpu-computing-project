#pragma once

#include "hittable_list.cuh"
#include "sphere.cuh"
#include "camera.cuh"
#include "material.cuh"

__global__ void g_createWorld(Hittable** d_list, Hittable** d_world, Camera** d_camera, int width, int height, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list) = new Sphere(Point3(0, 0, -1), 0.5, new Lambertian(Color(0.1, 0.2, 0.5)));
        *(d_list + 1) = new Sphere(Point3(0, -100.5, -1), 100, new Lambertian(Color(0.8, 0.8, 0.0)));
        *(d_list + 2) = new Sphere(Point3(1, 0, -1), 0.5, new Metal(Color(0.8, 0.6, 0.2), 0.3));
        *(d_list + 3) = new Sphere(Point3(-1, 0, -1), 0.5, new Dielectric(1.5));
        *(d_list + 4) = new Sphere(Point3(-1, 0, -1), -0.45, new Dielectric(1.5));
        *d_world = new HittableList(d_list, 5);

        Point3 lookfrom(3, 3, 2);
        Point3 lookat(0, 0, -1);
        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = 2.0;
        *d_camera = new Camera(lookfrom, lookat, Vec3(0, 1, 0), 20, float(width) / float(height), aperture, dist_to_focus);
    }
}

void createWorld(Hittable** list, Hittable** world, Camera** camera, int widht, int weight) {
    *(list) = new Sphere(Point3(0, 0, -1), 0.5, new Lambertian(Color(0.1, 0.2, 0.5)));
    *(list + 1) = new Sphere(Point3(0, -100.5, -1), 100, new Lambertian(Color(0.8, 0.8, 0.0)));
    *(list + 2) = new Sphere(Point3(1, 0, -1), 0.5, new Metal(Color(0.8, 0.6, 0.2), 0.3));
    *(list + 3) = new Sphere(Point3(-1, 0, -1), 0.5, new Dielectric(1.5));
    *(list + 4) = new Sphere(Point3(-1, 0, -1), -0.45, new Dielectric(1.5));
    *world = new HittableList(list, 5);

    Point3 lookfrom(3, 3, 2);
    Point3 lookat(0, 0, -1);
    float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 2.0;
    *camera = new Camera(lookfrom, lookat, Vec3(0, 1, 0), 20, float(widht) / float(weight), aperture, dist_to_focus);
}

__global__ void g_freeWorld(Hittable** d_list, Hittable** d_world, Camera** d_camera) {
    for (int i = 0; i < 5; i++) {
        delete ((Sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

void freeWorld(Hittable** list, Hittable** world, Camera** camera) {
    for (int i = 0; i < 5; i++) {
        delete ((Sphere*)list[i])->mat_ptr;
        delete list[i];
    }
    delete *world;
    delete *camera;
}