#pragma once

#include "hittable.cuh"
#include "random.cuh"
#include "ray.cuh"

class Material {
  public:
    __device__ virtual bool scatter(curandState_t *randState, const Ray &rayIn,
                                    const HitRecord &rec, Color &attenuation,
                                    Ray &scattered) const = 0;
    virtual bool scatter(const Ray &rayIn, const HitRecord &rec,
                         Color &attenuation, Ray &scattered) const = 0;
};

class Lambertian : public Material {
  public:
    __host__ __device__ Lambertian(const Color &a) : albedo(a) {}

    __device__ virtual bool scatter(curandState_t *randState, const Ray &rayIn,
                                    const HitRecord &rec, Color &attenuation,
                                    Ray &scattered) const override {
        Vec3 scatterDirection = rec.normal + randUnitVector(randState);
        if (scatterDirection.nearZero()) {
            scatterDirection = rec.normal;
        }
        scattered = Ray(rec.p, unitVector(scatterDirection));
        attenuation = albedo;
        return true;
    }

    virtual bool scatter(const Ray &rayIn, const HitRecord &rec,
                         Color &attenuation, Ray &scattered) const override {
        Vec3 scatterDirection = rec.normal + randUnitVector();
        if (scatterDirection.nearZero()) {
            scatterDirection = rec.normal;
        }
        scattered = Ray(rec.p, unitVector(scatterDirection));
        attenuation = albedo;
        return true;
    }

  private:
    Color albedo;
};

class Metal : public Material {
  public:
    __host__ __device__ Metal(const Color &a, float f)
        : albedo(a), fuzz(f < 1 ? f : 1) {}

    __device__ virtual bool scatter(curandState_t *randState, const Ray &rayIn,
                                    const HitRecord &rec, Color &attenuation,
                                    Ray &scattered) const override {
        Vec3 reflected = reflect(unitVector(rayIn.direction()), rec.normal);
        scattered = Ray(rec.p, reflected + fuzz * randInUnitSphere(randState));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

    virtual bool scatter(const Ray &rayIn, const HitRecord &rec,
                         Color &attenuation, Ray &scattered) const override {
        Vec3 reflected = reflect(unitVector(rayIn.direction()), rec.normal);
        scattered = Ray(rec.p, reflected + fuzz * randInUnitSphere());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

  private:
    Color albedo;
    float fuzz;
};

class Dielectric : public Material {
  public:
    __host__ __device__ Dielectric(float index_of_refraction)
        : ir(index_of_refraction) {}

    __device__ virtual bool scatter(curandState_t *randState, const Ray &rayIn,
                                    const HitRecord &rec, Color &attenuation,
                                    Ray &scattered) const override {
        attenuation = Color(1.0, 1.0, 1.0);
        float refractionRatio = rec.frontFace ? (1.0 / ir) : ir;

        Vec3 unitDirection = unitVector(rayIn.direction());
        float cosTheta = min(dot(-unitDirection, rec.normal), 1.0);
        float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

        bool cannotRefract = refractionRatio * sinTheta > 1.0;
        Vec3 direction;

        if (cannotRefract ||
            reflectance(cosTheta, refractionRatio) > randf(randState)) {
            direction = reflect(unitDirection, rec.normal);
        } else {
            refract(unitDirection, rec.normal, refractionRatio, direction);
        }

        scattered = Ray(rec.p, direction);
        return true;
    }

    virtual bool scatter(const Ray &rayIn, const HitRecord &rec,
                         Color &attenuation, Ray &scattered) const override {
        attenuation = Color(1.0, 1.0, 1.0);
        float refractionRatio = rec.frontFace ? (1.0 / ir) : ir;

        Vec3 unitDirection = unitVector(rayIn.direction());
        float cosTheta = fmin(dot(-unitDirection, rec.normal), 1.0);
        float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

        bool cannotRefract = refractionRatio * sinTheta > 1.0;
        Vec3 direction;

        if (cannotRefract || reflectance(cosTheta, refractionRatio) > randf()) {
            direction = reflect(unitDirection, rec.normal);
        } else {
            refract(unitDirection, rec.normal, refractionRatio, direction);
        }

        scattered = Ray(rec.p, direction);
        return true;
    }

  private:
    float ir; // Index of Refraction

    __host__ __device__ static float reflectance(float cosine, float ref_idx) {
        // Use Schlick's approximation for reflectance.
        float r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};