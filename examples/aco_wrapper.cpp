#include <cmath>
#include "dm.h"
#include "da.h"
#include "muesli.h"
#include <algorithm>
#include "array.h"
#include "Randoms.h"
#ifdef __CUDACC__
#define MSL_MANAGED __managed__
#define MSL_CONSTANT __constant__
#define POW(a, b)      powf(a, b)
#define EXP(a)      exp(a)
__device__ double d_PHERINIT;
__device__ double d_EVAPORATION;
__device__ double d_TAUMAX;
__device__ int d_BLOCK_SIZE;
__device__ int d_GRAPH_SIZE;
#else
#define MSL_MANAGED
#define MSL_CONSTANT
#define POW(a, b)      std::pow(a, b)
#define EXP(a)      std::exp(a)
#endif
#include <random>
typedef array<double, 2> city;
#define TAUMAX 2
#define IROULETE 32
#define CHECKCORRECTNESS 1
std::ostream& operator<< (std::ostream& os, const city t) {
    os << "(" << t[0] << ", " << t[1] << ")";
    return os;
}
namespace msl::aco {
    class Fill : public Functor<int, int> {
        int d;
    public:
        explicit Fill(int d) : d(d) {}
        MSL_USERFUNC int operator()(int x) const override {
            return d;
        }
    };
    class EtaTauCalc : public Functor5<int, int, double, double, double, double> {
    public:
        MSL_USERFUNC double operator()(int xindex, int yindex, double eta_tau, double dist, double phero) const override {
            int fromcity = xindex;
            int next_city = yindex;
            eta_tau = 0;
            if (fromcity != next_city) {
                double eta = pow(1/dist, 2);
                double tau = pow(phero, 1);
                eta_tau = eta * tau;
            }
            return eta_tau;
        }
    };
    class Min : public Functor2<double, double, double> {
    public:
        MSL_USERFUNC double operator()(double x, double y) const override {
            if (x < y) { return x; }
            return y;}
    };
    class CalcDistance : public Functor3<int, int, double, double> {
    private:
        city *cities;
    public:
        explicit CalcDistance(city * cities) {
            this->cities = cities;
        }
        MSL_USERFUNC double operator()(int cityi, int cityj, double value) const override {
            if (cityi == cityj) {
                return 0.0;
            } else {
                return sqrt(pow(cities[cityj][0] - cities[cityi][0], 2) +
                            pow(cities[cityj][1] - cities[cityi][1], 2));;
            }
        }
    };
    class TourConstruction: public Functor3<int, int *, double, double> {
    private:
        int width;
        int seed;
        int * iroulette{};
        double * etataus{};
        double * distances{};
    public:
        TourConstruction(int width, int seed) : width(width), seed(seed) {
        }
        void setIterationParams(int * _iroulette, double * _etataus, double * _distances) {
            this->iroulette = _iroulette;
            this->etataus = _etataus;
            this->distances = _distances;
        }
        MSL_USERFUNC double operator()(int ant_index, int * tour, double current_distance) const override {
            MSL_RANDOM_STATE randomState = msl::generateRandomState(this->seed, ant_index);
            int fromCity = msl::randInt(0, width - 1, randomState);
            tour[fromCity] = 0;
            double distance = 0;
            for (int i = 1; i < width; i++) {
                int nextCity = -1;
                double etaTauSum = 0;
                for (int j = 0; j < IROULETE; j++) {
                    int toCity = iroulette[fromCity * IROULETE + j];
                    if (tour[toCity] == -1) {
                        etaTauSum += etataus[toCity * width + fromCity];
                    }
                }
                if (etaTauSum != 0) {
                    double rand = msl::randDouble(0.0, etaTauSum, randomState);
                    double etaTauSum2 = 0;
                    for (int j = 0; j < IROULETE; j++) {
                        nextCity = iroulette[fromCity * IROULETE + j];
                        if (tour[nextCity] == -1) {
                            etaTauSum2 += etataus[nextCity * width + fromCity];
                        }
                        if (rand < etaTauSum2)
                            break;
                    }
                } else {
                    int startCity = msl::randInt(0, width - 1, randomState);
                    for (int j = 0; j < width; j++) {
                        if (tour[(startCity + j) % width] == -1) {
                            nextCity = (startCity + j) % width;
                            break;
                        }
                    }
                }
                tour[nextCity] = i;
                distance += distances[nextCity * width + fromCity];
                fromCity = nextCity;
            }
            return distance;
        }
    };
    class UpdateDelta : public Functor3<int, int, double, double> {
    private:
        int *flipped_routes{};
        double *dist_routes{};
        int nants, ncities;
    public:
        UpdateDelta(int cities, int ants) {
            this->nants = ants;
            this->ncities = cities;
        }
        void setIterationsParams(int *tours, double *distroutes) {
            this->flipped_routes = tours;
            this->dist_routes = distroutes;
        }
        MSL_USERFUNC double operator()(int row, int column, double phero) const override {
            int Q = 11340;
            double result = 0.0;
            for (int k = 0; k < nants; k++) {
                double rlength = dist_routes[k];
                int city1VisitIndex = flipped_routes[k * ncities + row];
                int city2VisitIndex = flipped_routes[k * ncities + column];
                if (abs(city1VisitIndex - city2VisitIndex) == 1) {
                    result += Q / rlength;
                }
            }
            return result;
        }
    };
    class UpdatePhero : public Functor4<int, int, double, double, double> {
    public:
        MSL_USERFUNC double operator()(int x, int y, double prevphero, double deltaphero) const override {
            double RO = 0.5;
            return (1 - RO) * (prevphero + deltaphero);
        }
    };
    void createPheroMatrix(int ncities, DM<double>& phero) {
        Randoms randoms(15);
        for (int j = 0; j < ncities; j++) {
            for (int k = 0; k <= j; k++) {
                if (j != k) {
                    double randn = randoms.Uniforme() * TAUMAX;
                    phero.set2D(j, k, randn);
                    phero.set2D(k, j, randn);
                }
            }
        }
        phero.updateDevice(true);
    }
    DM<int> createIRoulette(const DM<double> &distances, int ncities) {
        DM<int> iroulette(ncities, IROULETE, 0);
        for (int i = 0; i < ncities; i++) {
            for (int y = 0; y < IROULETE; y++) {
                double maxdistance = std::numeric_limits<double>::infinity();
                int city = -1;
                for (int j = 0; j < ncities; j++) {
                    bool check = true;
                    for (int k = 0; k < y; k++) {
                        if (iroulette.localPartition[i * IROULETE + k] == j) {
                            check = false;
                        }
                    }
                    if (i != j && check) {
                        double c_dist = distances.localPartition[i * ncities + j];
                        if (c_dist < maxdistance) {
                            maxdistance = c_dist;
                            city = j;
                        }
                    }
                }
                iroulette.set2D(i, y, city);
            }
        }
        iroulette.updateDevice(true);
        return iroulette;
    }
    void aco(int iterations, const std::string& importFile, int nants) {
        DA<city> cityArray(ncities, {});
        DM<double> phero(ncities, ncities, 0);
        createPheroMatrix(ncities, phero);
        DM<double> distance(ncities, ncities, 0);
        CalcDistance calcdistance(cityArray.getUserFunctionData());
        distance.mapIndexInPlace(calcdistance);
        distance.updateHost();
        DM<int> iroulette = createIRoulette(distance, ncities);
        DM<int> flipped_tours(nants, ncities, -1);
        DM<double> deltaphero(ncities, ncities, 0);
        DM<double> etatau(ncities, ncities, 0);
        DA<double> dist_routes(nants, 0);
        Fill fill(-1);
        Min min;
        double minroute = 0.0;
        EtaTauCalc etataucalc;
        int veryGoodSeed = (int) time(nullptr);
        TourConstruction tourConstruction(ncities, veryGoodSeed);
        UpdateDelta updatedelta(ncities, nants);
        UpdatePhero updatephero;
        double alltimeminroute = std::numeric_limits<double>::infinity();
        for (int i = 0; i < iterations; i++) {
            flipped_tours.mapInPlace(fill);
            etatau.zipIndexInPlace3(distance, phero, etataucalc);
            tourConstruction.setIterationParams(iroulette.getUserFunctionData(),
                                                etatau.getUserFunctionData(),
                                                distance.getUserFunctionData());
            dist_routes.mapIndexInPlaceDMRows(flipped_tours, tourConstruction);
            minroute = dist_routes.foldCPU(min);
            if (minroute < alltimeminroute) {
                alltimeminroute = minroute;
            }
            updatedelta.setIterationsParams(flipped_tours.getUserFunctionData(),
                                            dist_routes.getUserFunctionData());
            deltaphero.mapIndexInPlace(updatedelta);
            phero.zipIndexInPlace(deltaphero, updatephero);
        }
        dist_routes.updateHost();
        flipped_tours.updateHost();
    }
} // close namespaces