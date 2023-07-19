#pragma once

#include "hittable.cuh"

class Sphere : public Hittable {
  public:
    __host__ __device__ Sphere() {}
    __host__ __device__ Sphere(Point3 cen, float r, Material *m)
        : center(cen), radius(r), mat_ptr(m){};

    __host__ __device__ virtual bool hit(const Ray &r, float t_min, float t_max,
                                         HitRecord &rec) const override;

    friend std::ostream &operator<<(std::ostream &out, const Sphere &s);

  public:
    Point3 center;
    float radius;
    Material *mat_ptr;
};

__host__ __device__ bool Sphere::hit(const Ray &r, float t_min, float t_max,
                                     HitRecord &rec) const {
    Vec3 oc = r.origin() - center;
    float a = r.direction().squaredLength();
    float half_b = dot(oc, r.direction());
    float c = oc.squaredLength() - radius * radius;

    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0)
        return false;
    float sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    float root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    Vec3 outward_normal = (rec.p - center) / radius;
    rec.setFaceNormal(r, outward_normal);
    rec.material = mat_ptr;

    return true;
}

std::ostream &operator<<(std::ostream &out, const Sphere &s) {
    out << "Sphere(center: " << s.center << ", radius: " << s.radius << ")";
    return out;
}