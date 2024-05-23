#ifndef NBODY_ARRAY_H
#define NBODY_ARRAY_H

template<typename T, unsigned int N>
struct particle {
    float x, y, z, vx, vy, vz, mass, charge;

    MSL_USERFUNC T operator[](size_t n) const {
        return data[n];
    }

    MSL_USERFUNC T& operator[](size_t n) {
        return data[n];
    }
};

#endif //GASSIMULATION_ARRAY_H
