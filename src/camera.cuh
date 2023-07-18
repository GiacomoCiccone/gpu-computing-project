#pragma once

#include "random.cuh"
#include "ray.cuh"

#define DEG_TO_RAD(degrees) ((degrees)*M_PI / 180.0f)

class Camera {
  public:
    __host__ __device__ Camera(Point3 lookfrom, Point3 lookat, Vec3 vup,
                               float vfov, float aspect_ratio, float aperture,
                               float focus_dist) {
        float theta = DEG_TO_RAD(vfov);
        float h = tan(theta / 2);
        float viewport_height = 2.0f * h;
        float viewport_width = aspect_ratio * viewport_height;

        w = unitVector(lookfrom - lookat);
        u = unitVector(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner =
            origin - horizontal / 2 - vertical / 2 - focus_dist * w;

        lens_radius = aperture / 2;
    }

    Ray getRay(float s, float t) const {
        Vec3 rd = lens_radius * randInUnitDisk();
        Vec3 offset = u * rd.x() + v * rd.y();

        return Ray(origin + offset, lower_left_corner + s * horizontal +
                                        t * vertical - origin - offset);
    }

    __device__ Ray getRay(curandState *local_rand_state, float s,
                          float t) const {                            
        Vec3 rd = lens_radius * randInUnitDisk(local_rand_state);
        Vec3 offset = u * rd.x() + v * rd.y();

        return Ray(origin + offset, lower_left_corner + s * horizontal +
                                        t * vertical - origin - offset);
    }

  private:
    Point3 origin;
    Point3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 u, v, w;
    float lens_radius;
};