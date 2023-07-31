#pragma once

#include <curand_kernel.h>

#include "vec3.cuh"

// Questo file contiene funzioni per generare numeri casuali
// sia sulla CPU che sulla GPU
// Le interfacce sono le stesse, ma le funzioni per la GPU hanno
// un parametro aggiuntivo che è lo stato del generatore di numeri del thread
// corrente

float randf() { return (float)rand() / RAND_MAX; }

__device__ float randf(curandState *state) { return curand_uniform(state); }

float randf(float min, float max) { return min + (max - min) * randf(); }

__device__ float randf(float min, float max, curandState *state) {
    return min + (max - min) * randf(state);
}

Vec3 randVec3() { return Vec3(randf(), randf(), randf()); }

__device__ Vec3 randVec3(curandState *state) {
    return Vec3(randf(state), randf(state), randf(state));
}

Vec3 randVec3(float min, float max) {
    return Vec3(randf(min, max), randf(min, max), randf(min, max));
}

__device__ Vec3 randVec3(float min, float max, curandState *state) {
    return Vec3(randf(min, max, state), randf(min, max, state),
                randf(min, max, state));
}

Vec3 randInUnitSphere() {
    while (true) {
        Vec3 p = randVec3(-1, 1);
        if (p.squaredLength() >= 1)
            continue;
        return p;
    }
}

__device__ Vec3 randInUnitSphere(curandState *state) {
    while (true) {
        Vec3 p = randVec3(-1, 1, state);
        if (p.squaredLength() >= 1)
            continue;
        return p;
    }
}

Vec3 randInUnitDisk() {
    while (true) {
        Vec3 p = Vec3(randf(-1, 1), randf(-1, 1), 0);
        if (p.squaredLength() >= 1)
            continue;
        return p;
    }
}

__device__ Vec3 randInUnitDisk(curandState *state) {
    while (true) {
        Vec3 p = Vec3(randf(-1, 1, state), randf(-1, 1, state), 0);
        if (p.squaredLength() >= 1)
            continue;
        return p;
    }
}

Vec3 randUnitVector() { return unitVector(randInUnitSphere()); }

__device__ Vec3 randUnitVector(curandState *state) {
    return unitVector(randInUnitSphere(state));
}
