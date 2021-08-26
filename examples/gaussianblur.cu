/**
 * Copyright (c) 2020 Nina Herrmann
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */
#include <algorithm>
#include "dm.h"
#include "muesli.h"

#define EPSILON 0.03
#define MAX_ITER 1000
#ifdef __CUDACC__
#define POW(a, b)      powf(a, b)
#define EXP(a)      exp(a)
#else
#define POW(a, b)      std::pow(a, b)
#define EXP(a)      std::exp(a)
#endif
int rows, cols;
int* input_image_int;
char* input_image_char;
bool ascii = false;

namespace msl {

    namespace jacobi {


        int readPGM(const std::string& filename, int& rows, int& cols, int& max_color)
        {
            std::ifstream ifs(filename, std::ios::binary);
            if (!ifs) {
                std::cout << "Error: Cannot open image file " << filename << "!" << std::endl;
                return 1;
            }
            // Read magic number.
            std::string magic;
            getline(ifs, magic);
            if (magic.compare("P5")) { // P5 is magic number for pgm binary format.
                if (magic.compare("P2")) { // P2 is magic number for pgm ascii format.
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
                input_image_int = new int[rows*cols];
                int i = 0;
                while (getline(ifs, inputLine)) {
                    std::stringstream(inputLine) >> input_image_int[i++];
                }
            } else {
                input_image_char = new char[rows*cols];
                ifs.read(input_image_char, rows*cols);
            }
            return 0;
        }

        int writePGM(const std::string& filename, int * out_image, int rows, int cols, int max_color)
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
            ofs << "P5\n" << cols << " " << rows << " " << std::endl << max_color << std::endl;

            // Write image
            for (int x = 0; x < rows; x++) {
                for (int y = 0; y < cols; y++) {
                    unsigned char intensity = static_cast<unsigned char> (out_image[x*cols + y]);
                    ofs << intensity;
                }
            }

            if (ofs.fail()) {
                std::cout << "Cannot write file " << filename << "!" << std::endl;
                return 1;
            }

            return 0;
        }

        class GoLNeutralValueFunctor : public Functor2<int, int, int> {
        public:
            GoLNeutralValueFunctor(int default_neutral)
                    : default_neutral(default_neutral) {}

            MSL_USERFUNC
            int operator()(int x, int y) const {
                // All Border are not populated.
                return default_neutral;
            }

        private:
            int default_neutral = 0;
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
            int operator() (int row, int col, PLMatrix<int> *input, int ncol, int nrow) const
            {
                int offset = kw/2;
                float weight = 1.0f;
                float sigma = 1;
                float mean = (float)kw/2;

                // Convolution
                float sum = 0;
                for (int r = 0; r < kw; ++r) {
                    for (int c = 0; c < kw; ++c) {
                        sum += input->get(row+r-offset, col+c-offset) *
                                EXP(-0.5 * (POW((r-mean)/sigma, 2.0) + POW((c-mean)/sigma,2.0))) / (2 * M_PI * sigma * sigma);

                    }
                }
                return (int)sum/weight;
            }

        private:
            int kw = 10;
        };


        int testGaussian(std::string in_file, std::string out_file, int kw, bool output, int tile_width, int iterations, int iterations_used, std::string file, bool shared_mem) {
            int max_color;
            double start_init = MPI_Wtime();

            // Read image
            readPGM(in_file, rows, cols, max_color);
            DM<int> gs_image(rows, cols, 0, true);
            DM<int> gs_image_result(rows, cols, 0, true);
            if (ascii) {
                for (int i = 0; i < rows*cols; i++) {
                    gs_image.set(i,input_image_int[i]);
                }
            } else {
                for (int i = 0; i < rows*cols; i++) {
                    gs_image.set(i,input_image_char[i] - '0');
                }
            }
            double end_init = MPI_Wtime();
            if (msl::isRootProcess()) {
                if (output) {
                    std::ofstream outputFile;
                    outputFile.open(file, std::ios_base::app);
                    outputFile << "" << (end_init-start_init) << ";";
                    printf("%.2f;", end_init-start_init);
                    outputFile.close();
                }
            }
            double start = MPI_Wtime();

            Gaussian g(kw);
            g.setStencilSize(kw/2);
            g.setTileWidth(tile_width);
            g.setSharedMemory(shared_mem);
            GoLNeutralValueFunctor dead_nvf(0);
            for (int run = 0; run < iterations; ++run) {
                // Create distributed matrix to store the grey scale image.
                gs_image.mapStencilMM(gs_image_result, g, dead_nvf);
                gs_image_result.mapStencilMM(gs_image, g, dead_nvf);
            }
            double end = MPI_Wtime();
            if (msl::isRootProcess()) {
                if (output) {
                    std::ofstream outputFile;
                    outputFile.open(file, std::ios_base::app);
                    outputFile << "" << (end-start) << ";";
                    printf("%.2f;", end-start);
                    outputFile.close();
                }
            }
            gs_image_result.download();
            int *b = new int[rows*cols];
            b = gs_image_result.gather();
            if (msl::isRootProcess()) {
                if (output) {
                    writePGM(out_file, b, rows, cols, max_color);
                }
            }
            return 0;
        }

    } // namespace jacobi
} // namespace msl

int init(int row, int col)
{
    if (ascii) return input_image_int[row*cols+col];
    else return input_image_char[row*cols+col];
}
int main(int argc, char **argv) {
    std::cout << "\n\n************* Starting the Gaussian Blur *************\n ";

    msl::initSkeletons(argc, argv);
    int nGPUs = 1;
    int nRuns = 1;
    int iterations = MAX_ITER;
    int tile_width = msl::DEFAULT_TILE_WIDTH;
    msl::Muesli::cpu_fraction = 0.0;
    //bool warmup = false;
    bool output = false;
    bool shared_mem = false;
    int kw = 2;

    std::string in_file, out_file, file, nextfile;
    file = "result_travel.csv";
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
            shared_mem = true;
        }
        kw = atoi(argv[7]);
    }
    std::string shared = shared_mem ? "SM" : "GM";

    if (argc == 9) {
        in_file = argv[9];
        size_t pos = in_file.find(".");
        out_file = in_file;
        std::stringstream ss;
        ss << "_" << msl::Muesli::num_total_procs << "_" << nGPUs << "_" << iterations << "_" << shared <<  "_" << tile_width << "_" << kw << "_gaussian";
        out_file.insert(pos, ss.str());
    } else {
        in_file = "lena.pgm";
        std::stringstream oo;
        oo << in_file << "_" << msl::Muesli::num_total_procs << "_" << nGPUs << "_" << iterations << "_" << shared <<  "_" << tile_width << "_" << kw <<"_gaussian.pgm";
        out_file = oo.str();
    }
    output = true;
    std::stringstream ss;
    ss << file << "_" << iterations;
    nextfile = ss.str();
    msl::setNumGpus(nGPUs);
    msl::setNumRuns(nRuns);
    msl::setDebug(true);

    int iterations_used=0;
    for (int r = 0; r < msl::Muesli::num_runs; ++r) {
        msl::jacobi::testGaussian(in_file, out_file, kw, output, tile_width, iterations, iterations_used, nextfile, shared_mem);
    }

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
    std::cout << "\n************* Finished the Gaussian Blur *************\n ";

    return 0;
}
