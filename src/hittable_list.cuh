#pragma once

#include "hittable.cuh"

// Classe che rappresenta una lista di oggetti Hittable e semplifica la
// ricerca dell'oggetto più vicino al raggio
class HittableList : public Hittable {
  public:
    __host__ __device__ HittableList() {}
    __host__ __device__ HittableList(Hittable **l, int n) {
        list = l;
        list_size = n;
    }

    __host__ __device__ virtual bool hit(const Ray &r, float t_min, float t_max,
                                         HitRecord &rec) const override;

  private:
    Hittable **list;
    int list_size;
};

__host__ __device__ bool HittableList::hit(const Ray &r, float t_min,
                                           float t_max, HitRecord &rec) const {
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}