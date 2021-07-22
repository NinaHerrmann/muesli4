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

        int writePGM(const std::string& filename, DM<int>& out_image, int rows, int cols, int max_color)
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
                    unsigned char intensity = static_cast<unsigned char> (img[x][y]);
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
            Gaussian() : DMMapStencilFunctor(){}

            MSL_USERFUNC
            int operator() (int row, int col, PLMatrix<int> *input, int ncol, int nrow) const
            {
                int kw = 10;
                int offset = kw/2;
                float weight = 1.0f;
                float sigma = 1;
                float mean = (float)kw/2;

                // Convolution
                int sum = 0;
                for (int r = 0; r < kw; ++r) {
                    for (int c = 0; c < kw; ++c) {
                        sum += input->get(row+r-offset, col+c-offset) *
                               EXP(-0.5 * (POW((r-mean)/sigma, 2.0) + POW((c-mean)/sigma,2.0))) / (2 * M_PI * sigma * sigma);
                    }
                }

                return (int)sum/weight;
            }
        };


//         msl::jacobi::testGaussian(in_file, out_file, kw, output, tile_width, iterations, iterations_used);
        int testGaussian(std::string in_file, std::string out_file, int kw, bool output, int tile_width, int iterations, int iterations_used) {
            int max_color;
            double gauss_time = 0.0, t_upload = 0.0, t_padding =  0.0, t_kernel = 0.0;

            // Read image
            readPGM(in_file, rows, cols, max_color);
            msl::startTiming();
            for (int run = 0; run < Muesli::num_runs; ++run) {
                // Create distributed matrix to store the grey scale image.
                DM<int> gs_image(rows, cols, 1);
                DM<int> gs_image_result(rows, cols, 1);
                //writePGM("original.pgm", gs_image, rows, cols, max_color);

                double t = MPI_Wtime();
                // Gaussian blur
                //Gaussian g(kw);
                Gaussian g;

                g.setStencilSize(1);
                g.setTileWidth(tile_width);
                GoLNeutralValueFunctor dead_nvf(0);

                gs_image.mapStencilMM(gs_image_result, g, dead_nvf);
                //writePGM("afterGaussian.pgm", gs_image, rows, cols, max_color);

                // timing
                gauss_time += MPI_Wtime() - t;
                gauss_time = gauss_time;
            /*    t_upload += gs_image.getStencilTimes()[0];
                t_padding +=  gs_image.getStencilTimes()[1];
                t_kernel += gs_image.getStencilTimes()[2];*/

                if (output && msl::isRootProcess())
                    writePGM(out_file, gs_image, rows, cols, max_color);

                msl::splitTime(run);
            }
            msl::stopTiming();
            if (msl::isRootProcess()) {
                std::cout << "Gaussian time: " << gauss_time/Muesli::num_runs << std::endl
                          << "Upload time: " << t_upload/Muesli::num_runs << std::endl
                          << "Kernel time: " << t_kernel/Muesli::num_runs << std::endl
                          << "Padding time " << t_padding/Muesli::num_runs << std::endl;
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
    msl::initSkeletons(argc, argv);
    int nGPUs = 1;
    int nRuns = 1;
    int iterations = MAX_ITER;
    int tile_width = msl::DEFAULT_TILE_WIDTH;
    msl::Muesli::cpu_fraction = 0.0;
    //bool warmup = false;
    bool output = false;

    std::string in_file, out_file, file; //int kw = 10;
    file = "result.csv";
    if (argc >= 6) {
        nGPUs = atoi(argv[1]);
        nRuns = atoi(argv[2]);
        msl::Muesli::cpu_fraction = atof(argv[3]);
        if (msl::Muesli::cpu_fraction > 1) {
            msl::Muesli::cpu_fraction = 1;
        }
        tile_width = atoi(argv[4]);
        iterations = atoi(argv[5]);
    }
    if (argc == 7) {
        in_file = argv[8];
        size_t pos = in_file.find(".");
        out_file = in_file;
        out_file.insert(pos, "_gaussian");
    } else {
        in_file = "lena.pgm";
        out_file = "lena_gaussian.pgm";
        output = true;
        printf("I will take lena\n");
    }

    msl::setNumGpus(nGPUs);
    msl::setNumRuns(nRuns);

    int iterations_used=0;
    for (int r = 0; r < msl::Muesli::num_runs; ++r) {
        msl::jacobi::testGaussian(in_file, out_file, 10, output, tile_width, iterations, iterations_used);
    }

    if (output) {
/*        std::string id = "" + std::to_string(nGPUs) + ";" + std::to_string(tile_width) +";" + std::to_string(iterations) + ";" + std::to_string(iterations_used) +
                         ";" + std::to_string(msl::Muesli::cpu_fraction * 100) + ";\n";
        msl::printTimeToFile(id.c_str(), file);*/
        std::ofstream outputFile;
        outputFile.open(file, std::ios_base::app);
        outputFile << "" + std::to_string(nGPUs) + ";" + std::to_string(tile_width) +";" + std::to_string(iterations) + ";" + std::to_string(iterations_used) +
                      ";" + std::to_string(msl::Muesli::cpu_fraction * 100) + ";\n";
        outputFile.close();
    } else {
        msl::stopTiming();
    }
    msl::terminateSkeletons();
    return 0;
}
