#pragma once

#include "ray.cuh"

class Material;

struct HitRecord {
    float t;
    Point3 p;
    Vec3 normal;
    bool frontFace;
    Material *material;

    __host__ __device__ void setFaceNormal(const Ray &r,
                                           const Vec3 &outwardNormal) {
        frontFace = dot(r.direction(), outwardNormal) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
};

class Hittable {
  public:
    __host__ __device__ virtual bool hit(const Ray &r, float tMin, float tMax,
                                         HitRecord &rec) const = 0;
};