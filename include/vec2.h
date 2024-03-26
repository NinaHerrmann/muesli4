#ifndef ACO_VEC2_H
#define ACO_VEC2_H

template<typename T>
struct vec2 {

    T x, y;

    MSL_USERFUNC void operator+=(const vec2<T>& other) {
        x += other.x;
        y += other.y;
    }

    MSL_USERFUNC friend vec2<T> operator+(vec2<T> v, const vec2<T>& other) {
        v += other;
        return v;
    }

    MSL_USERFUNC void operator*=(const T& val) {
        x *= val;
        y *= val;
    }

    MSL_USERFUNC friend vec2<T> operator*(vec2<T> v, const T& val) {
        v *= val;
        return v;
    }

    MSL_USERFUNC friend T operator*(const vec2<T>& v1, const vec2<T>& v2) {
        return v1.x * v2.x + v1.y * v2.y;
    }
};

#endif //ACO_VEC2_H
