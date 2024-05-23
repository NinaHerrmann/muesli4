
/*
 * aco_flippedds.cpp
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
// 0 = space left, 1 = current_bin, 2 = #items
typedef array<int, 3> antstatus;

#define TAUMAX 2
#define IROULETE 32
#define CHECKCORRECTNESS 1

std::ostream &operator<<(std::ostream &os, const city t) {
    os << "(" << t[0] << ", " << t[1] << ")";
    return os;
}
// antfitness[0] = space left, antfitness[1] = current_bin, antfitness[2] = items_packed
std::ostream &operator<<(std::ostream &os, const antstatus t) {
    os << "(" << t[0] << "," << t[1] <<"," << t[2] << ")";
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

    class CalcProbs : public Functor6<int, int, double, antstatus, int, int, double> {
    private:
        const int diff_items, total_items;
        double *phero{};
        int *packed{};
    public:
        CalcProbs(int diff_items, int total_items) : diff_items(diff_items), total_items(total_items) {
        }

        void setOuterIterationsParams(double *_phero) {
            this->phero = _phero;
        }

        void setIterationParams(int *_packed) {
            this->packed = _packed;
        }

        MSL_USERFUNC double operator()(int antindex, int itemindex, double probs, antstatus antfitness, int item, int quantity) const override {
            if ((quantity == 0) || (antfitness[0] < item)) {
                return 0.0;
            } else {
                double pherosum = 0.0;
                for (int j = 0; j < total_items; j++) {
                    // In case the item is in the current bin, add up pheromone
                    if (packed[antindex * total_items + j] == antfitness[1]){
                        // Get the sum of all items which have been packed in the bin.
                        pherosum += phero[j * diff_items + itemindex];
                    }
                }
                // The weight increases the probability to take the item, as heavy objects should be packed first.
                return pherosum/antfitness[2] * (double) pow(item, 1);
            }
    }};
    class ReduceRow : public Functor2<double, double, double> {
    public:
        MSL_USERFUNC double operator()(double first, double second) const override {
            return first + second;
        }};

    class PackItem : public Functor4<int, int*, double, antstatus, antstatus> {
    private:
        const int diff_items;
        int seed, bin_capacity;
        int *items{};
        double *etatau{};
        int *bpp_quantity{};
    public:
        PackItem(int diff_items, int seed, int bin_capacity) : diff_items(diff_items), seed(seed), bin_capacity(bin_capacity){
        }

        void setIterationParams(int *_items, int *_bpp_quantity, double *_etatau){
            this->items = _items;
            this->bpp_quantity = _bpp_quantity;
            this->etatau = _etatau;
        }

        MSL_USERFUNC antstatus operator()(int ant_index, int* packed, double EtaTauSum, antstatus antfitness) const override {
            MSL_RANDOM_STATE randomState = msl::generateRandomState(this->seed, ant_index);
            // Change to "trueindex"; ++ quantity.
            int itemindex = 0;
            if (EtaTauSum != 0) {
                double rand = msl::randDouble(0.0, EtaTauSum, randomState);
                double etaTauSum2 = 0;
                // Add a random item to the bin - favored are items that are heavy and have a lot of pheromone.
                for (int j = 0; j < diff_items; j++) {
                    // the index of the item is sum of quantity.
                    itemindex += bpp_quantity[j];
                    if ((bpp_quantity[j] > 0) & (antfitness[0] > items[j])) {
                        etaTauSum2 += etatau[j];
                    }
                    if (rand < etaTauSum2) {
                        antfitness[0] -= items[j];
                        antfitness[2]++;
                        break;
                    }
                }
            } else {
                // We need a new bin.
                // Take the next heaviest item.
                int heaviest_object = 0.0;
                for (int j = 0; j < diff_items; j++) {
                     if ((items[j] > heaviest_object) & (bpp_quantity[j] > 0)) {
                         heaviest_object = items[j];
                     }
                 }
                 for (int j = 0; j < diff_items; j++) {
                     itemindex += bpp_quantity[j];
                     if (items[j] == heaviest_object) {
                         break;
                     }
                 }
                antfitness[0] = bin_capacity;
                antfitness[2] = 1;
                antfitness[1]++;
                antfitness[0] -= heaviest_object;
            }
            bpp_quantity[itemindex] -= 1.0;
            packed[itemindex] = antfitness[1];
            return antfitness;
        }
    };
    class BinPacking : public Functor3<int, int*, double, double> {
    private:
        const int diff_items;
        const int total_items;
        int seed;
        int *items{};
        double *phero{};
        int *bpp_quantity{};
        int heavyitem_index;
        int bin_capacity;
    public:
        BinPacking(int diff_items, int total_items, int seed, int heavyitem, int bin_capacity) : diff_items(diff_items),
        total_items(total_items), seed(seed), heavyitem_index(heavyitem), bin_capacity(bin_capacity){
        }

        void setIterationParams(int *_items, double *_phero, int *_bpp_quantity) {
            this->items = _items;
            this->phero = _phero;
            this->bpp_quantity = _bpp_quantity;
        }

        MSL_USERFUNC double operator()(int ant_index, int* packed, double antfitness) const override {
            MSL_RANDOM_STATE randomState = msl::generateRandomState(this->seed, ant_index);

            int bins_used = 0;
            // Change to "trueindex"; ++ quantity.
            int itemindex = 0;
            // Calculate SUM((BINWEIGHT/CAPACITY)^k) for favoring diverse bins in ant solutions.
            double sum_diversity_bins = 0.0;
            // Copy the quantity;
            int * bpp_quantity_cpy = new int[diff_items];
            for (int i = 0; i < diff_items; i++) {
                bpp_quantity_cpy[i] = bpp_quantity[i];
            }
            // Set the first item in the bin.
            int space_left = bin_capacity - items[heavyitem_index];
            bpp_quantity_cpy[heavyitem_index] -= 1.0;
            int items_in_bin = 1;
            packed[heavyitem_index] = bins_used;
            double * etatau = new double[diff_items];
            for (int i = 1; i < total_items; i++) {
                double etaTauSum = 0;
                double pherosum = 0.0;

                for (int j = 0; j < diff_items; j++) {
                    if ((bpp_quantity_cpy[j] > 0) & (space_left > items[j])) {
                        // get the sum of all items which hae been packed in the bin.
                        pherosum += phero[ i * diff_items + j];
                    }
                }
                for (int j = 0; j < diff_items; j++) {
                    if ((bpp_quantity_cpy[j] > 0) & (space_left > items[j])) {
                        // The weight increases the probability to take the item, as heavy objects should be packed first.
                        etatau[j] = pherosum/items_in_bin * (double) pow(items[j], 1);
                        etaTauSum += etatau[j];
                    }
                }

                if (etaTauSum != 0) {
                    double rand = msl::randDouble(0.0, etaTauSum, randomState);
                    double etaTauSum2 = 0;
                    // Add a random item to the bin - favored are items that are heavy and have a lot of pheromone.
                    for (int j = 0; j < diff_items; j++) {
                        // the index of the item is sum of quantity.
                        itemindex += bpp_quantity[j];
                        if ((bpp_quantity_cpy[j] > 0) & (space_left > items[j])) {
                            etaTauSum2 += etatau[j];
                        }
                        if (rand < etaTauSum2) {
                            space_left = space_left - items[j];
                            break;
                        }
                    }
                    items_in_bin++;
                } else {
                    // We need a new bin.
                    sum_diversity_bins += pow((bin_capacity - space_left) / bin_capacity, 2.0);
                    bins_used++;
                    // Take the next heaviest item.
                    int heaviest_object = 0.0;
                   /* for (int j = 0; j < diff_items; j++) {
                        if ((items[j] > heaviest_object) & (bpp_quantity_cpy[j] > 0)) {
                            heaviest_object = items[j];
                        }
                    }
                    for (int j = 0; j < diff_items; j++) {
                        itemindex += bpp_quantity[j];
                        if (items[j] == heaviest_object) {
                            break;
                        }
                    }*/
                    for (int j = 0; j < diff_items; j++) {
                        itemindex += bpp_quantity[j];
                        if ((bpp_quantity_cpy[j] > 0)) {
                            heaviest_object = items[j];
                            break;
                        }
                    }
                    space_left = bin_capacity - heaviest_object;
                    items_in_bin = 1;
                }
                bpp_quantity_cpy[itemindex] -= 1.0;
                packed[itemindex] = bins_used;
                itemindex = 0;
            }
            //printf("Bins used %f %d\n", sum_diversity_bins, bins_used);
            return (sum_diversity_bins)/bins_used;
        }
    };

    class UpdatePhero : public Functor3<int, int, double, double> {
    private:
        int *packed{};
        double *ant_fitness{};
        int width;
    public:
        UpdatePhero(int width) {
            this->width = width;
        }

        void setIterationsParams(int *_packed, double *_ant_fitness) {
            this->packed = _packed;
            this->ant_fitness = _ant_fitness;
        }
        MSL_USERFUNC double operator()(int x, int y, double prevphero) const override {
            double RO = 0.5;
            // Pheromone matrix displays if items of size x and y are likely to be packed in the same bin.
            // E.g. Bins of size 10 favor packing items auf size 6 and 4 together while bins of size 9
            // can not even pack that size.
            double deltaphero = 0.0;
            for (int k = 0; k < width; k++) {
                // If item x and y are packed in the same bin in the best solution the probability should increase.
                deltaphero += 32 / ant_fitness[k];
            }
            return (1 - RO) * (prevphero + deltaphero);
        }
    };
    class Min : public Functor2<double, double, double> {
    public:
        MSL_USERFUNC double operator()(double x, double y) const override {
            if (x < y) { return x; }
            return y;}
    };
    void readBPPFileProperties(const std::string &problem, int &n_objects_type, int &bin_capacity, bool is_palma) {
        std::ifstream fileReader;
        //Problem Instances
        std::string palma_path = "";
        if (is_palma) {
            palma_path = "/home/n/n_herr03/acomuesli/data/bpp/";
        } else {
            palma_path = "../data/bpp/";
        }

        fileReader.open(palma_path + problem + ".txt", std::ifstream::in);

        if (fileReader.is_open()) {
            fileReader >> n_objects_type;
            fileReader >> bin_capacity;
        }
        fileReader.close();
    }

    void readBPPFile(const std::string &problem, int n_objects_type, int &n_objects_total, int bin_capacity,
                     DA<int> &items, DA<int> &m_quantity, bool is_palma) {

        std::ifstream fileReader;

        std::string palma_path = "";
        if (is_palma) {
            palma_path = "/home/n/n_herr03/acomuesli/data/bpp/";
        } else {
            palma_path = "../data/bpp/";
        }

        //Problem Instances
        std::string file = problem + ".txt";

        fileReader.open(palma_path + file, std::ifstream::in);

        int lines = 0;
        double total = 0.0;

        if (fileReader.is_open()) {

            fileReader >> n_objects_type;
            fileReader >> bin_capacity;

            while (lines < n_objects_type && !fileReader.eof()) {
                double weight;
                double quantity;
                fileReader >> weight;
                fileReader >> quantity;
                items.setLocal(lines, weight);
                m_quantity.setLocal(lines, quantity);
                total += quantity;
                lines++;
            }
        } else {
            printf("\n File not opened");
        }

        n_objects_total = total;
        fileReader.close();
    }

    void createPheroMatrix(int n_objects, DM<double> &phero) {
        Randoms randoms(15);
        for (int j = 0; j < n_objects; j++) {
            for (int k = 0; k < n_objects; k++) {
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

    void bpp(int iterations, const std::string &importFile, int nants, int isPalma = 0) {
        msl::startTiming();
        bool is_palma = false;
        if (isPalma == 1) {
            is_palma = true;
        }
        int n_objects_type = 0;
        int bin_capacity = 0;
        int n_objects_total = 0;
        readBPPFileProperties(importFile, n_objects_type, bin_capacity, is_palma);
        DA<int> bpp_items(n_objects_type, 0);
        DA<int> bpp_qty(n_objects_type, 0);
        DA<double> antfitness(nants, 0);
        readBPPFile(importFile, n_objects_type, n_objects_total, bin_capacity, bpp_items, bpp_qty, is_palma);
        double best_fitness = std::numeric_limits<double>::infinity();

        DM<double> phero(n_objects_type, n_objects_type, 0);
        createPheroMatrix(n_objects_type, phero);
        DM<double> tau(n_objects_type, n_objects_type, 0);
        int maxweight = 0, max_weight_index = 0;
        for (int j = 0; j < n_objects_type; j++) {
            if (bpp_items.getLocal(j) > maxweight) {
                maxweight = bpp_items.getLocal(j);
            }
        }
        for (int j = 0; j < n_objects_type; j++) {
            max_weight_index += bpp_qty.getLocal(j);
            if (bpp_items.getLocal(j) == maxweight) {
                break;
            }
        }
        BinPacking binpacking(n_objects_type, n_objects_total, 0, max_weight_index, bin_capacity);
        UpdatePhero updatephero(n_objects_total);
        Min min;
        DM<int> flippeditems(nants, n_objects_total, 0);
        // TODO Max Int.
        int alltimeminbin = 99;
        DM<double> probs(n_objects_type, nants, 0);
        DA<antstatus> ds_antstatus(nants, {});
        DA<double> etatausum(nants, 0.0);

        CalcProbs calcprobs(n_objects_type, n_objects_total);
        ReduceRow reduceRow;
        PackItem packitem(n_objects_type, 0, bin_capacity);

        for (int i = 0; i < iterations; i++) {
            // Flip packing.
            /*binpacking.setIterationParams(bpp_items.getUserFunctionData(), phero.getUserFunctionData(),
                                          bpp_qty.getUserFunctionData());
            antfitness.mapIndexInPlaceDMRows(flippeditems, binpacking);*/
            calcprobs.setOuterIterationsParams(phero.getUserFunctionData());
            for (int j = 0; j < n_objects_total; j++) {
                calcprobs.setIterationParams(flippeditems.getUserFunctionData());
                probs.zipIndexInPlaceAAA(ds_antstatus, bpp_qty, bpp_items, calcprobs);
                probs.reduceRows(etatausum, reduceRow);
                packitem.setIterationParams(bpp_items.getUserFunctionData(), bpp_qty.getUserFunctionData(), probs.getUserFunctionData());
                ds_antstatus.zipIndexInPlaceDMRows(etatausum, flippeditems, packitem);
            }
            ds_antstatus.show("ds_antstatus");
            // Get the best route.
            // int minbin = ds_antstatus.foldCPU(min);
            for (int j = 0; j < nants; j++) {
                int numberbins = ds_antstatus.getLocal(j)[1];
                if (numberbins < alltimeminbin) {
                    alltimeminbin = numberbins;
                }
            }

            phero.mapIndexInPlace(updatephero);
        }
        msl::syncStreams();
        double calctime = msl::stopTiming();
        printf("%s;%d;%s;%f;%f;", importFile.c_str(), nants, "flippedds", calctime, best_fitness);
        msl::syncStreams();

        // msl::stopTiming();*/
    }
} // close namespaces

void exitWithUsage() {
    std::cerr
            << "Usage: ./gassimulation_test [-g <nGPUs>] [-n <iterations>] [-i <importFile>] [-t <threads>] [-c <cities>] [-a <ants>] [-r <runs>]"
            << "Default 1 GPU 1 Iteration No import File No Export File threads omp_get_max_threads cities 10 random generated cities ants 16 runs 1"
            << std::endl;
    exit(-1);
}

int getIntArg(char *s, bool allowZero = false) {
    int i = std::atoi(s);
    if (i < 0 || (i == 0 && !allowZero)) {
        exitWithUsage();
    }
    return i;
}

int main(int argc, char **argv) {
    msl::initSkeletons(argc, argv);
    int gpus = 1, iterations = 1, runs = 1, nants = 256, hpc = 0;
    std::string importFile = "";
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
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
            case 't':
                msl::setNumThreads(getIntArg(argv[i]));
                break;
            case 'h':
                hpc = getIntArg(argv[i], true);
                break;
            default:
                printf("entering default\n");
                exitWithUsage();
        }
    }
    msl::setNumGpus(gpus);
    msl::setNumRuns(runs);
    if (!importFile.empty()) {
        msl::aco::bpp(iterations, importFile, nants, hpc);
    } else {
        printf("Providing an import file is mandatory. \n");
        exit(-1);
    }

    msl::terminateSkeletons();
    return EXIT_SUCCESS;
}
