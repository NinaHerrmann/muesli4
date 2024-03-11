
/*
 * dm_test.cpp
 *
 *      Author: Nina Hermann,
 *  		Herbert Kuchen <kuchen@uni-muenster.de>
 *
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2020  Herbert Kuchen <kuchen@uni-muenster.de.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

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
__device__ double d_ALPHA;
__device__ double d_BETA ;
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
const int Q = 38;
typedef array<double, 2> city;
Randoms *randoms;


#define TAUMAX 2
#define IROULETE 32
#define CHECKCORRECTNESS 1

typedef array<int, IROULETE> tour;

std::ostream& operator<< (std::ostream& os, const city t) {
    os << "(" << t[0] << ", " << t[1] << ")";
    return os;
}std::ostream& operator<< (std::ostream& os, const tour t) {
    os << "(" << "";
    for (int i = 0; i < IROULETE; i++) {
        os << t[i] << " ";
    }
    os << ")";
    return os;
}
namespace msl::aco {

    class Reset : public Functor<double, double> {
    public:
        MSL_USERFUNC double operator()(double x) const override {
            return 0.0;
        }
    };
    class nextStep : public Functor2<int, tour, tour> {
    private:
        int * iroulette{};
        double * dist{};
        double * phero{};
        int ncities;
        double * randoms;
    public:
        nextStep(int ncities, double * randomp) {
            this->ncities = ncities;
            this->randoms = randomp;
        }
        void setIterationsParams(int * iroulet, double * pheromones, double * distances) {
            this->iroulette = iroulet;
            this->phero = pheromones;
            this->dist = distances;
        }

        MSL_USERFUNC tour operator()(int ant_index, tour citytobe) const override {
            auto * prob = new float [ncities];
            auto * eta = new float [ncities];
            auto * tau = new float [ncities];

            float sum;
            int d_ALPHA = 1;
            int d_BETA = 2;
            //start route steps

            for (int i=0; i < ncities-1; i++) {
                int cityi = citytobe[i];
                sum = 0.0f;

                for (int j = 0; j < ncities - 1; j++) {
                    if (cityi != j && !visited(j, i + 1, citytobe)) {
                        eta[j] = (float) pow(1 / dist[cityi * ncities + j], d_BETA);
                        tau[j] = (float) pow(phero[(cityi * ncities) + j], d_ALPHA);
                        sum += eta[j] * tau[j];
                    } else {
                        prob[j] = 0;
                    }
                }

                for (int j=0; j < ncities-1; j++) {
                    if (cityi != j && !visited(j, i+1, citytobe)) {
                        prob[j] = eta[j] * tau[j] / sum;
                    }
                }
                // choose next city
                int nextCity = city(ant_index, prob, randoms, IROULETE);

                if (nextCity < 0) {
                    int nc;
                    for (nc = 0; nc < ncities; nc++) {
                        if (!visited(nc, i+1, citytobe)) {
                            break;
                        }
                    }
                    nextCity = nc;
                }
                citytobe[(i + 1)] = nextCity;
            }
            return citytobe;
        }

        MSL_USERFUNC static int city(int antK, const float *probabilities, const double *rand_states, int iroulette) {
            double random = rand_states[antK];
            int i = 0;
            // In case a city was already visited the probability is zero therefore the while loop should not terminate.
            double sum = probabilities[0];
            while (sum < random) {
                i++;
                sum += probabilities[i];
            }
            if (i > iroulette) {
                return -1;
            }
            return (int) i;
        }

        MSL_USERFUNC static bool visited(int c, int step, tour citytobe) {
            for (int l=0; l <= step; l++) {
                if (citytobe[l] == c) {
                    return true;
                }
            }
            return false;
        }
    };

    class CalcProbs : public Functor5<int, int, double, double, double, double> {
    private:
        int i{}, ncities{};
        int * routes{}, * iroulette{};
    public:
        CalcProbs(int i, int ncities, int * iroulet) {
            this->i = i;
            this->ncities = ncities;
            this->iroulette = iroulet;
        }
        void setIterationsParams(int ii, int * tours) {
            this->i = ii;
            this->routes = tours;
        }

        MSL_USERFUNC double operator()(int x, int y, double t, double etatau, double sum) const override {
            // Calculates the probability of an ant going to the city at index x in the IROULETE closest cities.
            int cityi = routes[x*ncities+i];
            int next_city = iroulette[cityi*IROULETE+y];
            if (cityi == next_city || visited(x, next_city, routes, ncities, i+1)) {
                return 0;
            } else {
                if (sum == 0.0) {
                    return 0;
                } else {
                    return etatau / sum;
                }
            }
        }
        MSL_USERFUNC static bool visited(int antk, int c, const int* routes, int n_cities, int step) {
            for (int l=0; l <= step; l++) {
                if (routes[antk*n_cities+l] == c) {
                    return true;
                }
            }
            return false;
        }
    };

    class SUM : public Functor2<double, double, double> {
    public:
        MSL_USERFUNC double operator()(double x, double y) const override {
            return x + y;
        }
    };
    class Min : public Functor2<double, double, double> {
    public:
        MSL_USERFUNC double operator()(double x, double y) const override {
            if (x < y) { return x; }
            return y;}
    };
    class ZipSum : public Functor3<int, int, double, double> {
    private:
        double * dist{};
        int ncities{};
    public:
        explicit ZipSum(int cities) {
            this->ncities = cities;
        }
        void setDist(double * distance) {
            this->dist = distance;
        }
        MSL_USERFUNC double operator()(int x, int x2, double y) const override {
            double newdist = dist[x*ncities + x2];
            return y + newdist;
        }
    };
    class CalcRlength : public Functor2<int, double, double> {
    private:
        tour * routes{};
        double * distances{};
        int ncities;
    public:
        explicit CalcRlength(int cities) {
            this->ncities = cities;
        }
        void setIterationsParams(tour * tours, double * distance) {
            this->routes = tours;
            this->distances = distance;
        }
        MSL_USERFUNC double operator()(int x, double value) const override {
            double sum = 0.0;
            for (int j=0; j < ncities - 1; j++) {
                int cityi = routes[x][j];
                int cityj = routes[x][j + 1];
                sum += distances[cityi * ncities + cityj];
            }

            int cityi = routes[x][ncities - 1];
            int cityj = routes[x][0];
            sum += distances[cityi * ncities + cityj];

            return sum;
        }
    };
    class initialCity : public Functor3<int, int, int, int> {
    private:
        int *randoms;
    public:
        explicit initialCity(int * randomp) {
            this->randoms = randomp;
        }
        MSL_USERFUNC int operator()(int antindex, int city, int value) const override {
            if (city != 0) {
                return value;
            } else {
                return randoms[antindex];
            }
        }
    };

    class calcDistance : public Functor3<int, int, double, double> {
    private:
        city *cities;
    public:
        explicit calcDistance(city * cities) {
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
    class UpdateDelta : public Functor3<int, int, double, double> {
    private:
        tour *routes{};
        double *dist_routes{};
        int nants, ncities;
    public:
        UpdateDelta(int cities, int ants) {
            this->nants = ants;
            this->ncities = cities;
        }

        void setIterationsParams(tour *tours, double *distroutes) {
            this->routes = tours;
            this->dist_routes = distroutes;
        }

        MSL_USERFUNC double operator()(int row, int column, double phero) const override {
            int Q = 11340;
            int cityi = row;
            int cityj = column;
            double result = 0.0;
            for (int k = 0; k < nants; k++) {
                double rlength = dist_routes[k];
                dist_routes[k] = rlength;
                for (int r = 0; r < ncities-1; r++) {
                    if ((routes[k][r] == cityi && routes[k][r + 1] == cityj) ||
                            (routes[k][r] == cityj && routes[k][r + 1] == cityi)) {
                        result += Q / rlength;
                    }
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
    int readsize(const std::string &basicString) {
        int n_cities;
        if (basicString == "djibouti") {
            n_cities = 38;
        } else if (basicString == "luxembourg") {
            n_cities = 980;
        } else if (basicString == "catar") {
            n_cities = 194;
        } else if (basicString == "a280") {
            n_cities = 280;
        } else if (basicString == "d198") {
            n_cities = 198;
        } else if (basicString == "d1291") {
            n_cities = 1291;
        } else if (basicString == "lin318") {
            n_cities = 318;
        } else if (basicString == "pcb442") {
            n_cities = 442;
        } else if (basicString == "pcb1173") {
            n_cities = 1173;
        } else if (basicString == "pr1002") {
            n_cities = 1002;
        } else if (basicString == "pr2392") {
            n_cities = 2392;
        } else if (basicString == "rat783") {
            n_cities = 783;
        } else {
            std::cout << "No valid import file provided. Please provide a valid import file." << std::endl;
            exit(-1);
        }
        return n_cities;
    }
    void readData(const std::string &basicString, int ncities, DA <city> &cities, DM<double> &phero) {
        std::ifstream data;
        data.open("/home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/data/" + basicString + ".txt", std::ifstream::in);
        randoms = new Randoms(15);

        if (data.is_open()){
            double randn;
            for(int j = 0;j<ncities;j++){
                for(int k = 0;k<ncities;k++){
                    if(j!=k){
                        randn = randoms -> Uniforme() * TAUMAX;
                        phero.setLocal((j * ncities) + k, randn);
                        phero.setLocal((k * ncities) + j, randn);
                    }
                    else{
                        phero.setLocal((j * ncities) + k, 0.0);
                        phero.setLocal((k * ncities) + j, 0.0);
                    }
                }
            }
            int i = 0;
            double index, x, y;
            index = 0.0; x = 0.0; y = 0.0;
            city city = {0.0,0.0};
            while(i < ncities){
                data >> index;
                data >> x;
                data >> y;

                city[0] = (double)x;
                city[1] = (double)y;
                cities.setLocal(i, city);
                i += 1;
            }
            data.close();
        } else{
            printf(" File not opened\n");
        }
    }
    void checkminroute(int nants, double minroute, const DA<double>& dist_routes) {
        for (int ii = 0; ii < nants; ii++) {
            if (minroute > dist_routes.localPartition[ii]) {
                printf("minimum route %.2f bigger than distance at %d %.2f\n", minroute, ii, dist_routes.localPartition[ii]);
                exit(1);
            }
        }
    }
    bool check_ant(int ncities, const DA<tour>& routes, int ant) {
        bool returnvalid = true;
        tour antroute = routes.localPartition[ant];
        for (int jj = 0; jj < ncities; jj++) {
            int currentcity = antroute[jj];
            for (int kk = 0; kk < ncities; kk++) {
                if (kk != jj && currentcity == antroute[kk]) {
                    returnvalid = false;
                }
            }
        }
        return returnvalid;
    }
    void checkvalidroute(const DA<tour>& routes, int ncities, int nants) {
        int * invalidants = new int[nants];
        int j = 0;

        for (int ii = 0; ii < nants; ii++) {
            if (!check_ant(ncities, routes, ii)) {
                printf("Ant %d is invalid;", ii);
                invalidants[j] = ii;
                j++;
            }
        }
       /* for (int jj=0; jj <= j; jj++) {
            printf("[ ");
            for (int kk = 0; kk < ncities; kk++) {
                printf("%d ", routes.localPartition[invalidants[jj] * ncities + kk]);
            }
            printf("]\n");
        }*/
    }

    void aco(int iterations, const std::string& importFile, int nants) {
        int niroulet = IROULETE;
        int ncities = readsize(importFile);
        double ds2fill = 0.0;
        msl::startTiming();
        DA<city> cities(ncities, {});
        DM<double> phero(ncities, ncities, {});
        DM<double> distance(ncities, ncities, {});
        readData(importFile, ncities, cities, phero);
        DM<int> iroulet(ncities, niroulet, 0);
        calcDistance calcdistance(cities.getUserFunctionData());
        distance.mapIndexInPlace(calcdistance);
        distance.updateHost();
        for (int i = 0; i < ncities; i++) {
            for (int y = 0; y < IROULETE; y++) {
                double maxdistance = 999999.9;
                double c_dist;
                int city = -1;
                for (int j = 0; j < ncities; j++) {
                    bool check = true;
                    for (int k = 0; k < y; k++) {
                        if (iroulet.getLocal(i * IROULETE + k) == j) {
                            check = false;
                        }
                    }

                    if (i != j && check) {
                        c_dist = distance.getLocal(i*ncities + j);
                        if (c_dist < maxdistance) {
                            maxdistance = c_dist;
                            city = j;
                        }
                    }
                }
                iroulet.set(i * IROULETE + y, city);
            }
        }
        startTiming();
        msl::syncStreams();
        DA<tour> tours(nants, {});
        DM<double> deltaphero(ncities, ncities, 0);
        DA<double> dist_routes(nants, 0.0);
        DA<double> randompointer(nants, 0.0);
        for (int i = 0; i < nants; i++) {
            randompointer.set(i, randoms->Uniforme());
        }
        DA<int> randomstartcity(nants, 0);
        std::random_device rd;
        // Use Mersenne Twister engine for random number generation
        std::mt19937 gen(rd());
        // Define the distribution for integers between 1 and x
        std::uniform_int_distribution<> dis(1, ncities);
        for (int i = 0; i < nants; i++) {
            tours.set(i, {dis(gen)});
        }
        nextStep nextstep(ncities, randompointer.getUserFunctionData());
        initialCity initialcity(randomstartcity.getUserFunctionData());
        Reset reset;
        ZipSum zipsum(ncities);
        Min min;
        double minroute;
        zipsum.setDist(distance.getLocalPartition());
        UpdateDelta updatedelta(ncities, nants);
        UpdatePhero updatephero;
        CalcRlength calcrlength(ncities);
        ds2fill = msl::stopTiming();
       /* double etataucalctime = 0.0, reduceRowstime = 0.0, calcprobstime = 0.0, nextsteptime = 0.0, deltapherotime = 0.0,
        updatepherotime = 0.0, calcrlengthtime = 0.0, minroutetime = 0.0;*/
        double alltimeminroute = 999999.9;
        msl::startTiming();
        for (int i = 0; i < iterations; i++) {
            nextstep.setIterationsParams(iroulet.getUserFunctionData(), phero.getUserFunctionData(),
                                         distance.getUserFunctionData());
            tours.mapIndexInPlace(nextstep);
            tours.show("tours", 2);

            calcrlength.setIterationsParams(tours.getUserFunctionData(), distance.getUserFunctionData());
            // Calculate the length of the route.
            dist_routes.mapIndexInPlace(calcrlength);
            dist_routes.show("distroutes", 2);
            // Get the best route.
            //double minroutegpu = dist_routes.fold(min, true);
            minroute = dist_routes.foldCPU(min);
            //dist_routes.show();
            printf("minroute %f\n", minroute);
            if (minroute < alltimeminroute) {
                alltimeminroute = minroute;
            }
            updatedelta.setIterationsParams(tours.getUserFunctionData(), dist_routes.getUserFunctionData());
            // Calculate the delta pheromone.
            deltaphero.mapIndexInPlace(updatedelta);
            // Update the pheromone.
            phero.zipIndexInPlace(deltaphero, updatephero);
        }
        double calctime = msl::stopTiming();
        printf("%s;%d;%.6f;%.6f;\n", importFile.c_str(), nants, ds2fill, calctime);
               // %.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f - dsinit, fill, ds2fill, etataucalctime, reduceRowstime, calcprobstime, nextsteptime, deltapherotime, updatepherotime, calcrlengthtime, minroutetime, alltimeminroute);
        dist_routes.updateHost();
        tours.updateHost();
        if (CHECKCORRECTNESS) {
            checkminroute(nants, minroute, dist_routes);
            //checkvalidroute(tours, ncities, nants);
            printf("Made it!\n");
        }
        msl::stopTiming();
    }
} // close namespaces

void exitWithUsage() {
    std::cerr
            << "Usage: ./gassimulation_test [-g <nGPUs>] [-n <iterations>] [-i <importFile>] [-e <exportFile>] [-t <threads>] [-c <cities>] [-a <ants>] [-r <runs>]"
            << "Default 1 GPU 1 Iteration No import File No Export File threads omp_get_max_threads cities 10 random generated cities ants 16 runs 1" <<std::endl;
    exit(-1);
}

int getIntArg(char *s, bool allowZero = false) {
    int i = std::atoi(s);
    if (i < 0 || (i == 0 && !allowZero)) {
        printf("get Int arg %d %d\n", i, allowZero);
        exitWithUsage();
    }
    return i;
}

int main(int argc, char **argv) {
    msl::initSkeletons(argc, argv);
    // TODO does not work for less
    int gpus = 1, iterations = 1, runs = 1, nants = 256;
    std::string importFile, exportFile;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            printf("entering ! = - \n");
            exitWithUsage();
        }

        switch (argv[i++][1]) {
            case 'g':
                gpus = getIntArg(argv[i]);
                break;
            case 'a':
                nants = getIntArg(argv[i]);
                break;
            case 'n':
                iterations = getIntArg(argv[i], true);
                break;
            case 'i':
                importFile = std::string(argv[i]);
                break;
            case 'e':
                exportFile = std::string(argv[i]);
                break;
            case 't':
                msl::setNumThreads(getIntArg(argv[i]));
                break;
            case 'r':
                runs = (getIntArg(argv[i]));
                break;
            default:
                printf("entering default\n");
                exitWithUsage();
        }
    }
    msl::setNumGpus(gpus);
    msl::setNumRuns(runs);
    if (!importFile.empty()) {
        msl::aco::aco(iterations, importFile, nants);
    } else {
        printf("Providing an import file is mandatory. \n");
        exit(-1);
    }
    if (!exportFile.empty()) {
        // TODO Export result
    }
    msl::terminateSkeletons();
    return EXIT_SUCCESS;
}
