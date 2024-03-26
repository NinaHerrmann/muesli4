#ifndef MUESLI_TOUR_H
#define MUESLI_TOUR_H

template<typename T, unsigned int N>
struct tour {
    T data[N];
    double distance{};

    MSL_USERFUNC T operator[](size_t n) const {
        return data[n];
    }
    MSL_USERFUNC T& operator[](size_t n) {
        return data[n];
    }
    MSL_USERFUNC void setDist(double _distance) {
        distance = _distance;
    }
    MSL_USERFUNC double getDist() {
        return distance;
    };
    std::ostream& operator<< (std::ostream& os) {
        os << "(" << data[0] << ", " << data[1] << ")";
        return os;
    }
};

#endif //MUESLI_TOUR_H
