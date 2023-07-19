#pragma once

#include "vec3.cuh"

class Ray {
  public:
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Point3 &a, const Vec3 &b) {
        A = a;
        B = b;
    }
    __host__ __device__ inline Point3 origin() const { return A; }
    __host__ __device__ inline Vec3 direction() const { return B; }
    __host__ __device__ inline Point3 at(float t) const { return A + t * B; }

  private:
    Point3 A;
    Vec3 B;
};

inline std::ostream &operator<<(std::ostream &os, const Ray &r) {
    os << "Ray(" << r.origin() << ", " << r.direction() << ")";
    return os;
}

__host__ __device__ inline Vec3 reflect(const Vec3 &v, const Vec3 &n) {
    return v - 2 * dot(v, n) * n;
}

__host__ __device__ inline bool refract(const Vec3 &v, const Vec3 &n,
                                        float ni_over_nt, Vec3 &refracted) {
    Vec3 uv = unitVector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
        return true;
    } else {
        return false;
    }
}