#pragma once

#include "ray.cuh"

class Material;

// Contiene dati relativi all'intersezione di un raggio con un oggetto
struct HitRecord {
    float t;            // Distanza dal raggio
    Point3 p;           // Punto di intersezione
    Vec3 normal;        // Normale al punto di intersezione
    bool frontFace;     // Indica se la normale è orientata verso il raggio
    Material *material; // Materiale dell'oggetto colpito

    // Questa funzione serve per assicurarsi che la normale sia sempre
    // orientata verso il raggio che ha colpito l'oggetto
    __host__ __device__ void setFaceNormal(const Ray &r,
                                           const Vec3 &outwardNormal) {
        frontFace = dot(r.direction(), outwardNormal) < 0.0f;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
};

// Classe base per tutti gli oggetti che possono essere colpiti da un raggio
class Hittable {
  public:
    // Controlla se il raggio r interseca l'oggetto e, se si, restituisce
    // true e riempie il record di intersezione rec con i dati relativi
    // all'intersezione
    // tMin e tMax sono i valori di t per cui il raggio è valido
    __host__ __device__ virtual bool hit(const Ray &r, float tMin, float tMax,
                                         HitRecord &rec) const = 0;
};