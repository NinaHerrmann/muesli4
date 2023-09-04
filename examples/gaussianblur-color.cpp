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

#include "array.h"
#include "vec3.h"
#define EPSILON 0.03
#define MAX_ITER 50
#ifdef __CUDACC__
#define POW(a, b)      powf(a, b)
#define EXP(a)      exp(a)
#else
#define POW(a, b)      std::pow(a, b)
#define EXP(a)      std::exp(a)
#endif
int rows, cols;
array<int, 3>* input_image_int;
bool ascii = true;
typedef struct {
    int green;
    int blue;
    int red;
} colorpoint;

typedef array<int, 3> arraycolorpoint;

std::ostream& operator<< (std::ostream& os, const arraycolorpoint f) {
    os << "(" << f[0] << ", " << f[1] << ", " << f[2] << ")";
    return os;
}
namespace msl::gaussiancolor {

        int readPPM(const std::string& filename, int& rows, int& cols, int& max_color) {
            std::ifstream ifs(filename, std::ios::binary);
            if (!ifs) {
                std::cout << "Error: Cannot open image file " << filename << "!" << std::endl;
                return 1;
            }
            // Read magic number.
            std::string magic;
            getline(ifs, magic);
            if (magic.compare("P3")) { // P5 is magic number for pgm binary format.
                std::cout << "Image in PPM format \xE2\x9C\x93" << std::endl;
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
                input_image_int = new array<int, 3>[rows * cols];
                int i = 0;
                int j = 0;
                arraycolorpoint anarraycolorpoint = {0,0,0};
                while (getline(ifs, inputLine)) {
                    std::stringstream(inputLine) >> anarraycolorpoint[j++];
                    if (j == 3) {
                        j = 0;
                        input_image_int[i++] = anarraycolorpoint;
                        anarraycolorpoint = {0,0,0};
                    }
                }
            }
            return 0;
        }

        int writePPM(const std::string& filename, arraycolorpoint* out_image, int rows, int cols, int max_color) {
            std::ofstream ofs(filename, std::ios::binary);
            if (!ofs) {
                std::cout << "Error: Cannot open image file " << filename << "!" << std::endl;
                return 1;
            }
            // Gather full image
            int** img = new int*[rows];
            for (int i = 0; i < rows; i++)
                img[i] = new int[cols];

            // Write image header
            ofs << "P3\n" << cols << " " << rows << " " << std::endl << max_color << std::endl;

            // Write image
            for (int x = 0; x < rows; x++) {
                for (int y = 0; y < cols; y++) {
                    arraycolorpoint apoint = out_image[x * cols + y];
                    for (int j = 0; j < 3; j++) {
                        unsigned char intensity = static_cast<unsigned char>(apoint[j]);
                        ofs << apoint[j] << std::endl;
                    }
                }
            }

            if (ofs.fail()) {
                std::cout << "Cannot write file " << filename << "!" << std::endl;
                return 1;
            }

            return 0;
        }


/**
 * @brief Averages the top, bottom, left and right neighbours of a specific
 * element
 *
 */
        class Gaussian
                : public DMMapStencilFunctor<int, int> {
        public:
            Gaussian(int getkw): DMMapStencilFunctor() {
                kw = getkw;
            }

            MSL_USERFUNC
            array<int,3> operator() (int row, int col, PLMatrix<arraycolorpoint> *input, int ncol, int nrow) const
            {
                int offset = kw/2;
                float weight = 1.0f;
                float sigma = 1;
                float mean = (float)kw/2;
                // Convolution
                array<float, 3> sum = {0.0, 0.0, 0.0};
                for (int r = 0; r < kw; ++r) {
                    for (int c = 0; c < kw; ++c) {
                        sum[0] += input->get(row+r-offset, col+c-offset)[0] *
                                EXP(-0.5 * (POW((r-mean)/sigma, 2.0) + POW((c-mean)/sigma,2.0))) / (2 * M_PI * sigma * sigma);
                        sum[1] += input->get(row+r-offset, col+c-offset)[1] *
                                                        EXP(-0.5 * (POW((r-mean)/sigma, 2.0) + POW((c-mean)/sigma,2.0))) / (2 * M_PI * sigma * sigma);
                        sum[2] += input->get(row+r-offset, col+c-offset)[2] *
                                EXP(-0.5 * (POW((r-mean)/sigma, 2.0) + POW((c-mean)/sigma,2.0))) / (2 * M_PI * sigma * sigma);

                    }
                }
                array<int, 3> result = {(int)(sum[0]/weight), (int)(sum[1]/weight), (int)(sum[2]/weight)};
                return result;
            }

        private:
            int kw = 5;
        };


        double testGaussian(std::string in_file, const std::string& out_file, int kw, bool output, int iterations, const std::string& file) {
            int max_color;
            double start_init = MPI_Wtime();

            // Read image
            readPPM(in_file, rows, cols, max_color);
            array<int, 3> emptycolorpoint = {0,0,0};
            DM<arraycolorpoint> gs_image(rows, cols, emptycolorpoint);
            DM<arraycolorpoint> gs_image_result(rows, cols, emptycolorpoint);
            if (ascii) {
#pragma omp parallel for default(none) shared(gs_image, rows, cols, input_image_int)
                for (int i = 0; i < rows*cols; i++) {
                    gs_image.set(i, input_image_int[i]);
                }
                //gs_image.setPointer(input_image_int);
            }

            double end_init = MPI_Wtime();
            if (msl::isRootProcess()) {
                if (output) {
                    std::ofstream outputFile;
                    outputFile.open(file, std::ios_base::app);
                    outputFile << "" << (end_init-start_init) << ";";
                    outputFile.close();
                }
            }

            Gaussian g(5);
            g.setStencilSize(kw/2);
            g.setSharedMemory(false);
            msl::startTiming();

            for (int run = 0; run < iterations; ++run) {
                // Create distributed matrix to store the grey scale image.
                gs_image.mapStencilMM(gs_image_result, g, {0,0,0});
                gs_image_result.mapStencilMM(gs_image, g, {0,0,0});
            }
            double milliseconds = msl::stopTiming();

            if (msl::isRootProcess()) {
                if (output) {
                    std::ofstream outputFile;
                    outputFile.open(file, std::ios_base::app);
                    outputFile << "" << (milliseconds/1000) << ";";
                    outputFile.close();
                }
            }
            gs_image.updateHost();

            arraycolorpoint *b;
            b = gs_image.gather();
            if (msl::isRootProcess()) {
                if (output) {
		            std::ofstream outputFile;
                    outputFile.open(file, std::ios_base::app);
                    outputFile << "" << (milliseconds/1000) << ";\n";
                    outputFile.close();
                    writePPM(out_file, b, rows, cols, max_color);
                }
            }
	        return milliseconds;
        }
} // namespace msl


void exitWithUsage() {
    std::cerr << "Usage: ./gaussianblur [-g <nGPUs>] [-r <runs>] [-n <iterations>] [-i <importFiles>] [-t <number of threads>] [-k <radius of blur>]" << std::endl;
    exit(-1);
}

int getIntArg(char* s, bool allowZero = false) {
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
    int iterations = 1;
    int tile_width = msl::DEFAULT_TILE_WIDTH;
    msl::Muesli::cpu_fraction = 0.0;
    //bool warmup = false;
    bool output = true;
    int kw = 10;
    int reps = 1;
    msl::setReps(1);
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
        importFile = argv[9];
        size_t pos = importFile.find(".");
        out_file = importFile;
        std::stringstream ss;
        ss << "_" << msl::Muesli::num_total_procs << "_" << nGPUs << "_" << iterations <<  "_" << tile_width << "_" << kw << "_gaussian";
        out_file.insert(pos, ss.str());
    } else {
        importFile = "Data/4096x3072pexels.ppm";
        std::stringstream oo;
        oo << "Data/P_" << msl::Muesli::num_total_procs << "GPU_" << nGPUs << "I_" << iterations <<  "TW_" << tile_width << "R_" << reps << "KW_" << kw <<"_gaussian.ppm";
        out_file = oo.str();
    }
    output = true;
    std::stringstream ss;
    ss << file << "_" << iterations;
    nextfile = ss.str();
    msl::setNumRuns(nRuns);
    msl::setDebug(true);
    double miliseconds = 0;
    for (int r = 0; r < msl::Muesli::num_runs; ++r) {
        miliseconds = msl::gaussiancolor::testGaussian(importFile, out_file, kw, output, iterations, nextfile);
    }
    printf("Milliseconds: %.4f;", (miliseconds/1000/(float)msl::Muesli::num_runs));

    std::ofstream outputFile;
    outputFile.open(nextfile, std::ios_base::app);
    outputFile << "" + std::to_string(nGPUs) + ";" + std::to_string(tile_width) + ";" +
        std::to_string(iterations) + ";" + ";\n";
    outputFile.close();
    msl::terminateSkeletons();

    return 0;
}
