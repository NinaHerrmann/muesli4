// #include "../include/fss.h"
#include <iostream>
#include <time.h>
#include <memory>
#include <cmath>
#include <limits>
#include "muesli.h"
#include "dm.h"
#include "fssFunctors.cu"
//#include "darray.h"
//#include "lmatrix.h"
//#include "larray.h"
#include "rng.h"

namespace msl {
namespace fss {

//////////////////////////////////////////////////////////////////////////
//// constants
//////////////////////////////////////////////////////////////////////////
const double PI = 3.141592653589793;
const double EULER = 2.718281828459045;
const int multiplier= 1103515245;  // for random number generators
const int maxRand = 1<<31;         // for random number generators
const int increment = 12345;       // for random number generators

//////////////////////////////////////////////////////////////////////////
//// available fitness functions and corresponding functors
//////////////////////////////////////////////////////////////////////////
enum FitnessFunctions {
  ACKLEY,
  GRIEWANK,
  RASTRIGIN,
  ROSENBROCK,
  SCHWEFEL,
  SPHERE
};


////////////////////////////////////////////////////////////////////////
//// implementations of fitness functions
////////////////////////////////////////////////////////////////////////

class CalcFitness{
private:
  LMatrix<double> fishes_;
  size_t dimensions_;
  FitnessFunctions fitness_function_;
 
public:  
CalcFitness(DM<double>& fishes, size_t dimensions, FitnessFunctions fitness_function)
    : fishes_ (fishes ),
      dimensions_ (dimensions),
      fitness_function_(fitness_function) {
  this->addArgument(&fishes_);
}

double operator()(int i, double x) const {
  switch (fitness_function_) {
    case FitnessFunctions::ACKLEY:
      return ackley(i, x);
    case FitnessFunctions::GRIEWANK:
      return griewank(i, x);
    case FitnessFunctions::RASTRIGIN:
      return rastrigin(i, x);
    case FitnessFunctions::ROSENBROCK:
      return rosenbrock(i, x);
    case FitnessFunctions::SCHWEFEL:
      return schwefel(i, x);
    case FitnessFunctions::SPHERE:
      return sphere(i, x);
    default:
      return 0;
  }
}

double ackley(int i, double x) const {
  double sum1 = 0;
  double sum2 = 0;
  for (size_t j = 0; j < dimensions_; ++j) {
    auto value = fishes_.get(i, j);
    sum1 += std::pow(value, 2);
    sum2 += std::cos(2 * PI * value);
  }
  return -(-20 * std::exp(-0.2 * std::sqrt(1.0 / dimensions_ * sum1))
      - std::exp(1.0 / dimensions_ * sum2) + 20 + EULER);
}

double griewank(int i, double x) const {
  double sum = 0;
  double product = 1;
  for (size_t j = 1; j <= dimensions_; ++j) {
    double value = fishes_.get(i, j - 1);
    sum += std::pow(value, 2) / 4000;
    product *= std::cos(value / std::sqrt(static_cast<double>(j)));
  }
  return -(1 + sum - product);
}

double rastrigin(int i, double x) const {
  double sum = 0;
  for (size_t j = 0; j < dimensions_; ++j) {
    double value = fishes_.get(i, j);
    sum += (std::pow(value, 2) - 10 * std::cos(2 * PI * value));
  }
  return -(10 * dimensions_ + sum);
}

double rosenbrock(int i, double x) const {
  double sum = 0;
  for (size_t j = 0; j < dimensions_ - 1; ++j) {
    double value = fishes_.get(i, j);
    sum += (100 * std::pow((fishes_.get(i, j + 1) - std::pow(value, 2)), 2)
        + std::pow(1 - value, 2));
  }
  return -sum;
}

double schwefel(int i, double x) const {
  double outer_sum = 0;
  for (size_t j = 0; j < dimensions_; ++j) {
    double inner_sum = 0;
    for (size_t k = 0; k <= j; ++k) {
      inner_sum += std::pow(fishes_.get(i, k), 2);
    }
    outer_sum += inner_sum;
  }
  return -outer_sum;
}

double sphere(int i, double x) const {
  double sum = 0;
  for (size_t j = 0; j < dimensions_; ++j) {
    double value = fishes_.get(i, j);
    sum += (value * value);
  }
  return -sum;
}

////////////////////////////////////////////////////////////////////////
//// class Config
////////////////////////////////////////////////////////////////////////

class Config {
private: 
  size_t problem_dimensions_;
  double upper_bound_;
  
public:  

Config(){}

~Config() {
//  delete fitness_functor_;
}

void init(FitnessFunctions fitness_function, size_t problem_dimensions) {
  problem_dimensions_ = problem_dimensions;

  switch (fitness_function) {
    case FitnessFunctions::ACKLEY:
      upper_bound_ = 32;
      break;
    case FitnessFunctions::GRIEWANK:
      upper_bound_ = 600;
      break;
    case FitnessFunctions::RASTRIGIN:
      upper_bound_ = 5.12;
      break;
    case FitnessFunctions::ROSENBROCK:
      upper_bound_ = 30;
      break;
    case FitnessFunctions::SCHWEFEL:
      upper_bound_ = 100;
      break;
    case FitnessFunctions::SPHERE:
      upper_bound_ = 100;
      break;
    default:
      upper_bound_ = 0;
      break;
  }
}

size_t getProblemDimensions() const {
  return problem_dimensions_;
}

double getUpperBound() const {
  return upper_bound_;
}

double getLowerBound() const {
  return -getUpperBound();
}

double getProblemRange() const {
  return getUpperBound() - getLowerBound();
}

double getInitUpperBound() const {
  return getUpperBound();
}

double getInitLowerBound() const {
  return getLowerBound();
}

double getWeightUpperBound() const {
  return 5000;
}

double getWeightLowerBound() const {
  return 1;
}

double getStepSizeInitial() const {
  return 0.1;
}

double getStepSizeFinal() const {
  return 0.00001;
}

double getStepSizeVolitiveInitial() const {
  return 2 * getStepSizeInitial();
}

double getStepSizeVolitiveFinal() const {
  return 2 * getStepSizeFinal();
}

////////////////////////////////////////////////////////////////////////
//// Main FSS function
////////////////////////////////////////////////////////////////////////

double fss(size_t number_of_fishes, size_t dimensions, size_t iterations,
           FitnessFunctions fitness_function, bool output, bool progress) {

  // *** set up config **************************
  auto *conf = &Config::getInstance();
  conf->init(fitness_function, dimensions);

  // **** create data structures and variables  *****************
  DMatrix<double> fishes(number_of_fishes, dimensions);

  DMatrix<double> best_fishes(number_of_fishes, dimensions);

  // initial value for minimization problem
  double best_fitness = std::numericLimits<double>::max(); 
  double new_fitness = std::numericLimits<double>::max(); 

  DM<double> candidate_fishes(number_of_fishes, dimensions, Muesli::num_local_procs, 1);

  DM<double> fishes_last_iteration(number_of_fishes, dimensions,
                                   Muesli::num_local_procs, 1);

  DM<double> displacement(number_of_fishes, dimensions,
                               Muesli::num_local_procs, 1);

  DA<double> fitness(number_of_fishes, std::numeric_limits<double>::lowest());

  DArray<double> candidate_fitness(number_of_fishes,
                                   std::numeric_limits<double>::lowest()

  DA<double> last_fitness(number_of_fishes,
                          std::numeric_limits<double>::lowest());

  DA<double> fitness_variation(number_of_fishes,
                                   std::numeric_limits<double>::lowest());

  DA<double> weight(number_of_fishes, conf->getWeightLowerBound());

  DA<double> instinctive_movement_vector(dimensions);

  DA<double> instinctive_movement_vector_copy_dist(dimensions);

  DM<double> weighted_fishes(number_of_fishes, dimensions,
                                  Muesli::num_local_procs, 1);

  DA<double> barycenter(dimensions);

  DA<double> barycenter_copy_dist(dimensions);

  DA<double> distances_fishes_barycenter(number_of_fishes);

  // functors
  
  // CopyFishes copy_fishes_functor; 
  // UpdateFitness update_fitness;
  // CopyFitness copy_fitness;
  // MZipDif dif_zip_matrix_functor { };


// ***************************** start of main fss computation   *******************
  // init fish matrix
  InitFishes init_fish(conf->getInitLowerBound(), conf->getInitUpperBound());
  fishes.mapIndexInPlace(init_fish);

  // start iterations
  for (size_t current_iteration = 0; current_iteration < iterations;
      ++current_iteration) {

    // calculate fitness
    CalcFitness calc_fitness(dimensions, fitness_function);
    fitness = fishes.foldRows(calc_fitness);

    // store best results
    BestFitness calc_best_fitness;
    new_fitness = fitness.fold(calc_best_fitness,true);
    if (new_fitness < best_fitness) best_fitness = new_fitness;

    last_fitness.zipInPlace(fitness, [] (double x) {return x;});

/////////////////////////////////////////////////////////////////////////////////
// individual movement
    IndividualMovement individual_movement(
      conf->getLowerBound(), conf->getUpperBound(), conf->getStepSizeInitial(),
      conf->getStepSizeFinal(), iterations);
    candidate_fishes.zipInPlace(fishes,individual_movement);
    individual_movement.reduceStepSize();

    candidate_fitness = candidate_fishes.foldRows(calc_fitness);

    CopyCandidates copy_candidates;
    fishes.zipInPlace4(fitness,candidate_fitness,
                       candidate_fishes,copy_candidates);
    fitness = fishes.foldRows(calc_fitness);

    // store best results
    new_fitness = fitness.fold(calc_best_fitness,true);
    if (new_fitness < best_fitness) best_fitness = new_fitness;

/////////////////////////////////////////////////////////////////////////////////
// feeding; required for volitive movement
    CalcFitnessVariation calc_fitness_variation;
    fitness_variation.zipInPlace4(fitness,last_fitness,calc_fitness_variation);

    MaxArray calc_max_fitness_variation;
    double max_fitness_variation = fitness_variation.fold(
        calc_max_fitness_variation, true);

    Feeding feeding(conf->getWeightLowerBound(), 
                    conf->getWeightUpperBound());
    weight.zipInPlace(max_fitness_variation,feeding);

///////////////////////////////////////////////////////////////////////////////////
    // instinctive movement
    SumArray sum_array_functor;
    double sum_fitness_variation = fitness_variation.fold(sum_array_functor,true);
    
    CalcDisplacement calc_displacement_functor;
    displacement.zipInPlace3(fishes,fishes_last_iteration,calc_displacement);

    CalcDisplacementFitnessVector calc_displacement_fitness_vector;
    displacement.zipInPlace(fitness_variation,calc_displacement_fitness_vector);  // delta x * delta f

    SumMatrix sum_matrix_functor { };
    instinctive_movement_vector = displacement.foldCols(sum_matrix_functor);  // sum displacement * delta f

    CalcInstinctiveMovementVector calc_instinctive_movement_vector(sum_fitness_variation);
    instinctive_movement_vector.mapInPlace(calc_instinctive_movement_vector);  // divide sum by (sum fitness variation)
    instinctive_movement_vector.gather(instinctive_movement_vector_copy_dist);

    InstinctiveMovement instinctive_movement(conf->getLowerBound(), conf->getInitUpperBound());
    fishes.zipInPlace(instinctive_movement_vector,instinctive_movement);  // apply vector

    ///////////////////////////////////////////////////////////////////////////////
    // volitive movement

    double sum_weight = weight.fold(sum_array_functor, true);
    
    CalcWeightedFishes calc_weighted_fishes;
    weighted_fishes.zipInPlace3(fishes, weight,calc_weighted_fishes);

    barycenter = weighted_fishes.foldCols(sum_matrix_functor);

    CalcBarycenter calc_barycenter_functor(sum_weight);
    barycenter.mapInPlace(calc_barycenter_functor);
    barycenter.gather(barycenter_copy);

    CalcEuclideanDistance calc_euclidean_distance(barycenter_copy, dimensions);
    distances_fishes_barycenter.zipInPlace3(fishes,barycenter_copy,calc_euclidean_distance);

    VolitiveMovement volitive_movement(conf->getLowerBound(), conf->getUpperBound(),
      conf->getStepSizeVolitiveInitial(), conf->getStepSizeVolitiveFinal(),
      iterations, number_of_fishes);
    volitive_movement.nextIteration(sum_weight);
    fishes.zipInPlace(barycneter_copy, distances_fishes_barycenter,volitive_movement);
  }  // end of fss iteration loop

  ////////////////////////////////////////////////////////////////////////////////////////////////////
//  collect final result

  msl::printv("Best fitness found: %e\n", best_fitness);
}

}
}

////////////////////////////////////////////////////////////////////////
//// main function
////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  msl::initSkeletons(argc, argv);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  printf("Process %i runs on %s.\n", msl::Muesli::proc_id, processor_name);

  MPI_Barrier(MPI_COMM_WORLD);

  auto fitness_function = msl::fss::FitnessFunctions::SPHERE;
  auto fishes = 4;
  auto dimension = 2;
  auto iterations = 5;
  auto num_gpus = 2;
  auto runs = 1;

  bool output = false;
  bool progress = false;

  if (argc < 7) {
    if (msl::isRootProcess()) {
      std::cout << std::endl;
      std::cout
          << "Usage: "
          << argv[0]
          << " #nFishes #nDimension #nIterations #Fitness_Function #nRuns #nGPUs"
          << std::endl;
      std::cout << "Available fitness functions: " << std::endl << "0\tACKLEY"
                << std::endl << "1\tGRIEWANK" << std::endl << "2\tRASTRIGIN"
                << std::endl << "3\tROSENBROCK" << std::endl << "4\tSCHWEFEL"
                << std::endl << "5\tSPHERE" << std::endl;
      std::cout << std::endl << std::endl;
    }
    msl::terminateSkeletons();
    return EXIT_SUCCESS;
  } else {
    fishes = atoi(argv[1]);
    dimension = atoi(argv[2]);
    iterations = atoi(argv[3]);
    fitness_function = static_cast<msl::fss::FitnessFunctions>(atoi(argv[4]));
    runs = atoi(argv[5]);
    num_gpus = atoi(argv[6]);
  }

  msl::printv("runs = %i\n", runs);
  msl::printv("fishes = %i\n", fishes);
  msl::printv("dimensions = %i\n", dimension);
  msl::printv("iterations = %i\n", iterations);
  auto s_fitness_function = "";
  switch (fitness_function) {
    case 0:
      s_fitness_function = "Ackley";
      break;
    case 1:
      s_fitness_function = "Griewank";
      break;
    case 2:
      s_fitness_function = "Rastrigin";
      break;
    case 3:
      s_fitness_function = "Rosenbrock";
      break;
    case 4:
      s_fitness_function = "Schwefel";
      break;
    case 5:
      s_fitness_function = "Sphere";
      break;
  }
  msl::printv("Fitness function = %s\n\n", s_fitness_function);

  msl::setNumRuns(runs);
  msl::setNumGpus(num_gpus);

  msl::startTiming();

  double total_fitness = 0;
  for (int run = 0; run < msl::Muesli::num_runs; ++run) {
    total_fitness += msl::fss::fss(fishes, dimension, iterations,
                                   fitness_function, output, progress);
    msl::splitTime(run);
  }
  msl::stopTiming();

  msl::printv("Average fitness: %f\n", total_fitness / runs);

  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}
