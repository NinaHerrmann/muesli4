#ifndef GASSIMULATION_VEC3_H
#define GASSIMULATION_VEC3_H

template<typename T>
struct vec3 {

    T x, y, z;

    MSL_USERFUNC void operator+=(const vec3<T>& other) {
        x += other.x;
        y += other.y;
        z += other.z;
    }

    MSL_USERFUNC friend vec3<T> operator+(vec3<T> v, const vec3<T>& other) {
        v += other;
        return v;
    }

    MSL_USERFUNC void operator*=(const T& val) {
        x *= val;
        y *= val;
        z *= val;
    }

    MSL_USERFUNC friend vec3<T> operator*(vec3<T> v, const T& val) {
        v *= val;
        return v;
    }

    MSL_USERFUNC friend T operator*(const vec3<T>& v1, const vec3<T>& v2) {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }
};

#endif //GASSIMULATION_VEC3_H
