/**
 * Copyright (c) 2020 Nina Herrmann
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */
#include <algorithm>
#include "dm.h"
#include "muesli.h"
#include <iostream>

#define EPSILON 0.03
#define MAX_ITER 1
#ifdef __CUDACC__
#define POW(a, b)      powf(a, b)
#define EXP(a)      exp(a)
#else
#define POW(a, b)      std::pow(a, b)
#define EXP(a)      std::exp(a)
#endif
int rows, cols;
int *input_image_int;
char *input_image_char;
bool ascii = true;

namespace msl::gaussianblur {

    int readPGM(const std::string &filename, int &rows, int &cols, int &max_color) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) {
            std::cout << "Error: Cannot open image file " << filename << "!" << std::endl;
            return 1;
        }
        // Read magic number.
        std::string magic;
        getline(ifs, magic);
        if (magic == "P5") { // P5 is magic number for pgm binary format.
            if (magic == "P2") { // P2 is magic number for pgm ascii format.
                std::cout << "Error: Image not in PGM format!" << std::endl;
                return 1;
            }
            ascii = true;
        }

        // Skip comments
        std::string inputLine;
        while (true) {
            getline(ifs, inputLine);
            if (inputLine[0] != '#') break;
        }

        // Read image size and max color.
        std::stringstream(inputLine) >> cols >> rows;
        getline(ifs, inputLine);
        std::stringstream(inputLine) >> max_color;
        if (msl::isRootProcess()) {
            std::cout << "\nmax_color: " << max_color << "\t cols: " << cols << "\t rows: " << rows << std::endl;
        }
        // Read image.
        if (ascii) {
            input_image_int = new int[rows * cols];
            int i = 0;
            while (getline(ifs, inputLine)) {
                std::stringstream(inputLine) >> input_image_int[i++];
            }
        } else {
            input_image_char = new char[rows * cols];
            ifs.read(input_image_char, rows * cols);
        }

        return 0;
    }

    int writePGM(const std::string &filename, const int *out_image, int rows, int cols, int max_color) {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) {
            std::cout << "Error: Cannot open image file " << filename << "!" << std::endl;
            return 1;
        }
        // Gather full image
        int **img = new int *[rows];
        for (int i = 0; i < rows; i++)
            img[i] = new int[cols];

        // Write image header
        ofs << "P2\n" << cols << " " << rows << " " << std::endl << max_color << std::endl;

        // Write image
        for (int x = 0; x < rows; x++) {
            for (int y = 0; y < cols; y++) {
                auto intensity = static_cast<unsigned char> (out_image[x * cols + y]);
                ofs << out_image[x * cols + y] << std::endl;
            }
        }

        if (ofs.fail()) {
            std::cout << "Cannot write file " << filename << "!" << std::endl;
            return 1;
        }

        return 0;
    }

/**
 * @brief Averages the top, bottom, left and right borders of a specific element
 */
    class Gaussian
            : public DMMapStencilFunctor<int, int> {
    public:
        explicit Gaussian(int getkw) : DMMapStencilFunctor() {
            kw = getkw;
        }

        MSL_USERFUNC
        int operator()(int row, int col, PLMatrix<int> *input, int ncol, int nrow) const {
            int offset = kw / 2;
            double weight = 1.0f;
            float sigma = 1;
            float mean = (float) kw / 2;
            // Convolution
            double sum = 0;
            for (int r = 0; r < kw; ++r) {
                for (int c = 0; c < kw; ++c) {
                    sum += input->get(row + r - offset, col + c - offset) *
                           EXP(-0.5 * (POW((r - mean) / sigma, 2.0) + POW((c - mean) / sigma, 2.0))) /
                           (2 * M_PI * sigma * sigma);

                }
            }
            return (int) (sum / weight);
        }

    private:
        int kw = 5;
    };


    double testGaussian(const std::string &in_file, const std::string &out_file, int kw, int iterations,
                       const std::string &file) {
        int max_color;
        double start_init = MPI_Wtime();

        // Read image
        readPGM(in_file, rows, cols, max_color);
        DM<int> gs_image(rows, cols, 0, true);
        DM<int> gs_image_result(rows, cols, 0, true);
        if (ascii) {
            for (int i = 0; i < rows * cols; i++) {
                gs_image.set(i, input_image_int[i]);
            }
        } else {
            input_image_int = new int[rows * cols];
            for (int i = 0; i < rows * cols; i++) {
                input_image_int[i] = ((int) input_image_char[i]);
                if ((int) input_image_char[i] < 0) {
                    std::cout << "string:    " << input_image_char[i] << std::endl;
                }
            }
            for (int i = 0; i < rows * cols; i++) {
                gs_image.set(i, input_image_int[i]);
            }
            gs_image.updateDevice();
        }

        double end_init = MPI_Wtime();
        if (msl::isRootProcess()) {
            std::ofstream outputFile;
            outputFile.open(file, std::ios_base::app);
            outputFile << "" << (end_init - start_init) << ";";
            outputFile.close();
        }

        Gaussian g(kw);
        g.setStencilSize(kw / 2);
        g.setSharedMemory(false);
        msl::startTiming();

        for (int run = 0; run < iterations; ++run) {
            // Create distributed matrix to store the grey scale image.
            gs_image.mapStencilMM(gs_image_result, g, 0);
            gs_image_result.mapStencilMM(gs_image, g, 0);
        }
        double milliseconds = msl::stopTiming();

        if (msl::isRootProcess()) {
            std::ofstream outputFile;
            outputFile.open(file, std::ios_base::app);
            outputFile << "" << (milliseconds / 1000) << ";";
            outputFile.close();
        }
        gs_image_result.updateHost();
        int *b;
        b = gs_image_result.gather();
        if (msl::isRootProcess()) {
            std::ofstream outputFile;
            outputFile.open(file, std::ios_base::app);
            outputFile << "" << (milliseconds / 1000) << ";\n";
            outputFile.close();
            writePGM(out_file, b, rows, cols, max_color);

        }
        return milliseconds;
    }

} // namespace msl

int init(int row, int col) {
    if (ascii) return input_image_int[row * cols + col];
    else return input_image_char[row * cols + col];
}

void exitWithUsage() {
    std::cerr
            << "Usage: ./gaussianblur [-g <nGPUs>] [-r <runs>] [-n <iterations>] [-i <importFiles>] [-t <number of threads>] [-k <radius of blur>]"
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
    int nGPUs = 1;
    int nRuns = 1;
    int iterations = MAX_ITER;
    int tile_width = msl::DEFAULT_TILE_WIDTH;
    msl::Muesli::cpu_fraction = 0.0;
    int kw = 4;
    int reps = 1;

    std::string importFile, out_file, file, nextfile;
    file = "result.csv";

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            exitWithUsage();
        }
        switch (argv[i++][1]) {
            case 'g':
                msl::setNumGpus(getIntArg(argv[i]));
                break;
            case 'r':
                nRuns = getIntArg(argv[i], true);
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
            case 'k':
                kw = getIntArg(argv[i]);
                break;
            default:
                exitWithUsage();
        }
    }

    if (!importFile.empty()) {
        size_t pos = importFile.find(".");
        out_file = importFile;
        std::stringstream ss;
        ss << "_" << msl::Muesli::num_total_procs << "_" << nGPUs << "_" << iterations << "_" << kw
           << "_gaussian";
        out_file.insert(pos, ss.str());
    } else {
        importFile = "Data/4096x3072pexels-sapir.pgm";
        std::stringstream oo;
        oo << "Data/Sapir_" << "P_" << msl::Muesli::num_total_procs << "GPU_" << nGPUs << "I_" << iterations << "R_"
        << reps << "KW_" << kw << "_gaussian.pgm";
        out_file = oo.str();
    }
    std::stringstream ss;
    ss << file << "_" << iterations;
    nextfile = ss.str();
    msl::setNumRuns(nRuns);
    msl::setDebug(true);
    msl::setReps(reps);
    int iterations_used = 0;
    double milliseconds = 0.0f;
    for (int r = 0; r < msl::Muesli::num_runs; ++r) {
        milliseconds = msl::gaussianblur::testGaussian(importFile, out_file, kw, iterations, nextfile);
    }
    printf("%.2f;", (milliseconds / 1000 / (float) msl::Muesli::num_runs));

    std::ofstream outputFile;
    outputFile.open(nextfile, std::ios_base::app);
    outputFile
            << "" + std::to_string(nGPUs) + ";" + std::to_string(tile_width) + ";" + std::to_string(iterations) + ";" +
               std::to_string(iterations_used) + ";\n";
    outputFile.close();
    msl::terminateSkeletons();

    return 0;
}
