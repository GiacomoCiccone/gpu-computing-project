#pragma once

#include "vec3.cuh"

class Ray {
public:
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Vec3& a, const Vec3& b) { A = a; B = b; }
    __host__ __device__ inline Vec3 origin() const { return A; }
    __host__ __device__ inline Vec3 direction() const { return B; }
    __host__ __device__ inline Vec3 pointAt(float t) const { return A + t * B; }

private:
    Vec3 A;
    Vec3 B;
};

inline std::ostream& operator << (std::ostream& os, const Ray& r) {
    os << "Ray(" << r.origin() << ", " << r.direction() << ")";
    return os;
}

__host__ __device__ inline Ray reflect(const Ray& r, const Vec3& n) {
    return Ray(r.origin(), r.direction() - 2 * dot(r.direction(), n) * n);
}

__host__ __device__ inline bool refract(const Ray& r, const Vec3& n, float ni_over_nt, Ray& refracted) {
    Vec3 uv = unitVector(r.direction());
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = Ray(r.origin(), ni_over_nt * (uv - n * dt) - n * sqrt(discriminant));
        return true;
    }
    else {
        return false;
    }
}