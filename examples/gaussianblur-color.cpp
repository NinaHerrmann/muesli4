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
namespace msl {

    namespace jacobi {


        int readPNG(const std::string& filename, int& rows, int& cols, int& max_color)
        {
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

        int writePPM(const std::string& filename, arraycolorpoint* out_image, int rows, int cols, int max_color)
        {
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

        class GoLNeutralValueFunctor : public Functor2<int, int, array<int, 3>> {
        public:
            GoLNeutralValueFunctor(array<int, 3> default_neutral)
                    : default_neutral(default_neutral) {}

            MSL_USERFUNC
            array<int, 3> operator()(int x, int y) const {
                // All Border are not populated.
                return default_neutral;
            }

        private:
            array<int, 3> default_neutral = {0,0,0} ;
        };

/**
 * @brief Averages the top, bottom, left and right neighbours of a specific
 * element
 *
 */
        class Gaussian
                : public DMMapStencilFunctor<int, int, GoLNeutralValueFunctor> {
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


        float testGaussian(std::string in_file, std::string out_file, int kw, bool output, int tile_width, int iterations, int iterations_used, std::string file, bool shared_mem) {
            int max_color;
            double start_init = MPI_Wtime();

            // Read image
            readPNG(in_file, rows, cols, max_color);
            array<int, 3> emptycolorpoint = {0,0,0};
            printf("Nrows %d ncol %d\n", rows, cols);
            DM<arraycolorpoint> gs_image(rows, cols, emptycolorpoint);
            DM<arraycolorpoint> gs_image_result(rows, cols, emptycolorpoint);
            if (ascii) {
                for (int i = 0; i < rows*cols; i++) {
                    gs_image.set(i,input_image_int[i]);
                }
            }
            //gs_image.show();

            double end_init = MPI_Wtime();
            if (msl::isRootProcess()) {
                if (output) {
                    std::ofstream outputFile;
                    outputFile.open(file, std::ios_base::app);
                    outputFile << "" << (end_init-start_init) << ";";
                    outputFile.close();
                }
            }
            //double start = MPI_Wtime();

            Gaussian g(5);
            g.setStencilSize(kw/2);
            g.setSharedMemory(false);
            GoLNeutralValueFunctor dead_nvf({0,0,0});
            for (int run = 0; run < iterations; ++run) {
                // Create distributed matrix to store the grey scale image.
                gs_image.mapStencilMM(gs_image_result, g, dead_nvf);
                gs_image_result.mapStencilMM(gs_image, g, dead_nvf);
            }
            float milliseconds = 0;

            //double end = MPI_Wtime();

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

    } // namespace jacobi
} // namespace msl


int main(int argc, char **argv) {
    //std::cout << "\n\n************* Starting the Gaussian Blur *************\n ";

    msl::initSkeletons(argc, argv);
    int nGPUs = 1;
    int nRuns = 1;
    int iterations = MAX_ITER;
    int tile_width = msl::DEFAULT_TILE_WIDTH;
    msl::Muesli::cpu_fraction = 0.0;
    //bool warmup = false;
    bool output = true;
    bool shared_mem = false;
    int kw = 10;
    int reps = 1;

    std::string in_file, out_file, file, nextfile;
    file = "result_lena.csv";

    if (argc >= 7) {
        nGPUs = atoi(argv[1]);
        nRuns = atoi(argv[2]);
        msl::Muesli::cpu_fraction = atof(argv[3]);
        if (msl::Muesli::cpu_fraction > 1) {
            msl::Muesli::cpu_fraction = 1;
        }
        tile_width = atoi(argv[4]);
        iterations = atoi(argv[5]);
        if (atoi(argv[6]) == 1) {
            shared_mem = false;
        }
        kw = atoi(argv[7]);
        if (argc >= 9) {
            reps = atoi(argv[8]);
        }
    }
    std::string shared = shared_mem ? "SM" : "GM";

    if (argc == 10) {
        in_file = argv[9];
        size_t pos = in_file.find(".");
        out_file = in_file;
        std::stringstream ss;
        ss << "_" << msl::Muesli::num_total_procs << "_" << nGPUs << "_" << iterations << "_" << shared <<  "_" << tile_width << "_" << kw << "_gaussian";
        out_file.insert(pos, ss.str());
    } else {
        in_file = "Data/PokemonTT.ppm";
        //in_file = "lena.pgm";
        std::stringstream oo;
        oo << "Data/PTT_" << "P_" << msl::Muesli::num_total_procs << "GPU_" << nGPUs << "I_" << iterations << "_" << shared <<  "TW_" << tile_width << "R_" << reps << "KW_" << kw <<"_gaussian.ppm";
        out_file = oo.str();
    }
    output = true;
    std::stringstream ss;
    ss << file << "_" << iterations;
    nextfile = ss.str();
    msl::setNumGpus(nGPUs);
    msl::setNumRuns(nRuns);
    msl::setDebug(true);
    msl::setReps(reps);
    int iterations_used=0;
    printf("%d;%d;%d;", tile_width,kw,reps);
    float miliseconds = 0;
    for (int r = 0; r < msl::Muesli::num_runs; ++r) {
        miliseconds = msl::jacobi::testGaussian(in_file, out_file, kw, output, tile_width, iterations, iterations_used, nextfile, shared_mem);
    }
    printf("%.2f;", (miliseconds/1000/msl::Muesli::num_runs));

    if (output) {
        std::ofstream outputFile;
        outputFile.open(nextfile, std::ios_base::app);
        outputFile << "" + std::to_string(nGPUs) + ";" + std::to_string(tile_width) +";" + std::to_string(iterations) + ";" +
        std::to_string(iterations_used) + ";\n";
        outputFile.close();
    } else {
        msl::stopTiming();
    }
    msl::terminateSkeletons();

    return 0;
}
