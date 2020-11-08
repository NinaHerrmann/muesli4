// #include "../include/fss.h"
#include <iostream>
#include <time.h>
#include <memory>
#include <cmath>
#include <limits>
#include "muesli.h"
#include "dm.h"
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
//// FSS functors
////////////////////////////////////////////////////////////////////////

class InitFishes : public Functor<double, double> {
 private:
   Rng<double> rng_;
public:
  InitFishes(double min, double max)
      : rng_ { min, max } {
  }

  MSL_USERFUNC
  double operator()(double x) const {
    return rng_();
  }
};

class CopyFishes : public Functor<double, double> {
 private:
  LMatrix<double> fishes_;
 public:
  CopyFishes(DMatrix<double>& fishes)
      : fishes_(fishes) {
    this->addArgument(&fishes_);
  }

  MSL_USERFUNC
  double operator()(int i, int j, double x) const {
    return fishes_.get(i, j);
  }
};

class IndividualMovement : public Functor<double, double> {
private:
  LMatrix<double> fishes_;
  double value_min_, value_max_, step_size_initial_, step_size_final_, step_size_;
  size_t iteration_current_, iteration_max_;
  msl::Rng<double> rng_;
public:
  IndividualMovement(DMatrix<double>& fishes, double value_min,
                     double value_max, double step_size_initial,
                     double step_size_final, size_t iteration_max)
      : fishes_(fishes),
        value_min_ { value_min },
        value_max_ { value_max },
        step_size_initial_ { step_size_initial },
        step_size_final_ { step_size_final },
        step_size_ { step_size_initial },
        iteration_current_ { 0 },
        iteration_max_ { iteration_max },
        rng_ { -1, 1 } {
    this->addArgument(&fishes_);
  }

  void nextIteration() {
    ++iteration_current_;
    if (iteration_current_ > 1) {
      step_size_ = step_size_
          - ((step_size_initial_ - step_size_final_)
              / static_cast<double>(iteration_max_ - 1));
    }
  }

  MSL_USERFUNC
  double operator()(int i, int j, double x) const {
    double rand_factor = rng_();

    double direction = rand_factor * step_size_ * (value_max_ - value_min_);

    double new_x = fishes_.get(i, j) + direction;

    if (new_x < value_min_) {
      new_x = value_min_;
    } else if (new_x > value_max_) {
      new_x = value_max_;
    }

    return new_x;
  }
};

class CopyCandidates : public Functor<double, double> {
 private:
   LMatrix<double> candidates_;
   LArray<double> candidate_fitness_;
   LArray<double> fitness_;
 public:
  CopyCandidates(DMatrix<double>& candidates, DArray<double>& candidate_fitness,
                 DArray<double>& fitness)
      : candidates_(candidates),
        candidate_fitness_(candidate_fitness),
        fitness_(fitness) {
    this->addArgument(&candidates_);
    this->addArgument(&candidate_fitness_);
    this->addArgument(&fitness_);
  }

  MSL_USERFUNC
  double operator()(int i, int j, double x) const {
    if (candidate_fitness_.get(i) > fitness_.get(i)) {
      return candidates_.get(i, j);
    }
    return x;
  }
};

class UpdateFitness : public Functor<double, double> {
private:
  LArray<double> candidate_fitness_;
public:
  UpdateFitness(DArray<double>& candidate_fitness)
      : candidate_fitness_(candidate_fitness) {
    this->addArgument(&candidate_fitness_);
  }

  MSL_USERFUNC
  double operator()(int i, double x) const {
    if (candidate_fitness_.get(i) > x) {
      return candidate_fitness_.get(i);
    }
    return x;
  }
};

class CopyFitness : public Functor<double, double> {
private:
  LArray<double> fitness_;
public:
  CopyFitness(DArray<double>& fitness)
      : fitness_(fitness) {
    this->addArgument(&fitness_);
  }

  MSL_USERFUNC
  double operator()(int i, double x) const {
    return fitness_.get(i);
  }


};

class CalcFitnessVariation : public Functor<double, double> {
 private:
   LArray<double> fitness_;
   LArray<double> previous_fitness_;
 public:
  CalcFitnessVariation(DArray<double>& fitness, DArray<double>& previous_fitness)
      : fitness_(fitness),
        previous_fitness_(previous_fitness) {
    this->addArgument(&fitness_);
    this->addArgument(&previous_fitness_);
  }

  MSL_USERFUNC
  double operator()(int i, double x) const {
    return fitness_.get(i) - previous_fitness_.get(i);
  }
};

class MaxArray : public Functor2<double, double, double> {
public:
  MSL_USERFUNC
  double operator()(double a, double b) const {
    return (a >= b) ? a : b;
  }
};

class SumArray : public Functor2<double, double, double> {
public:
  MSL_USERFUNC
  double operator()(double a, double b) const {
    return a + b;
  }
};

class SumMatrix : public Functor2<double, double> {
public:
  MSL_USERFUNC
  double operator()(double a, double b) const {
    return a + b;
  }
};

class MZipDif : public Functor2<double, double, double> {
 public:
  MSL_USERFUNC
  double operator()(double a, double b) const {
    return a - b;
  }
};

class CalcDisplacement : public Functor<double, double> {
private:
   LMatrix<double> fishes_;
   LMatrix<double> fishes_last_iteration_;
 public:
  CalcDisplacement(DMatrix<double>& fishes,
                   DMatrix<double>& fishes_last_iteration)
      : fishes_(fishes),
        fishes_last_iteration_(fishes_last_iteration) {
    this->addArgument(&fishes_);
    this->addArgument(&fishes_last_iteration_);
  }

  MSL_USERFUNC
  double operator()(int i, int j, double x) const {
    return fishes_.get(i, j) - fishes_last_iteration_.get(i, j);
  }
};

class Feeding : public Functor<double, double> {
 private:
   LArray<double> fitness_variation_;
   double max_fitness_variation_;
   double weight_lower_bound_;
   double weight_upper_bound_;
 public:
  Feeding(DArray<double>& fitness_variation, double weight_lower_bound,
          double weight_upper_bound)
      : fitness_variation_(fitness_variation),
        max_fitness_variation_(0),
        weight_lower_bound_(weight_lower_bound),
        weight_upper_bound_(weight_upper_bound) {
    this->addArgument(&fitness_variation_);
  }

  void setMaxFitnessVariation(double max_fitness_variation) {
    max_fitness_variation_ = max_fitness_variation;
  }

  MSL_USERFUNC
  double operator()(int i, double x) const {
    if (max_fitness_variation_ < 1e-20)
      return x;

    auto result = x + fitness_variation_.get(i) / max_fitness_variation_;

    if (result > weight_upper_bound_) {
      result = weight_upper_bound_;
    } else if (result < weight_lower_bound_) {
      result = weight_lower_bound_;
    }

    return result;
  }
};

class CalcDisplacementFitnessVector : public Functor<double,double> {
private:
  LArray<double> fitness_variation_;
public:
  CalcDisplacementFitnessVector(DArray<double>& fitness_variation)
      : fitness_variation_(fitness_variation) {
    this->addArgument(&fitness_variation_);
  }

  MSL_USERFUNC
  double operator()(int i, int j, double x) const {
    return x * fitness_variation_.get(i);
  }
};

class CalcInstinctiveMovementVector : public Functor<double, double> {
private:
  double sum_fitness_variation_;
  
public:
  CalcInstinctiveMovementVector()
      : sum_fitness_variation_(0) {
  }

  void setSumFitnessVariation(double sum_fitness_variation) {
    sum_fitness_variation_ = sum_fitness_variation;
  }

  MSL_USERFUNC
  double operator()(double x) const {
    if (sum_fitness_variation_ < 1e-20) {
      return 0;
    }
    return x / sum_fitness_variation_;
  }
};

class InstinctiveMovement : public Functor2<double, double, double> {
private:
  LArray<double> instinctive_movement_vector_;
  double value_min_, value_max_;
  
public:
  InstinctiveMovement(DArray<double>& instinctive_movement_vector,
                      double value_min, double value_max)
      : instinctive_movement_vector_ { instinctive_movement_vector,
            Distribution::COPY },
        value_min_ { value_min },
        value_max_ { value_max } {
    this->addArgument(&instinctive_movement_vector_);
  }

  MSL_USERFUNC
  double operator()(int i, int j, double x) const {

    double val = instinctive_movement_vector_.get(j);
    double new_x = x + val;

    if (new_x < value_min_) {
      new_x = value_min_;
    } else if (new_x > value_max_) {
      new_x = value_max_;
    }

    return new_x;
  }
};

class CalcWeightedFishes : public Functor3<double, double, double, double> {
private:
  LMatrix<double> fishes_;
  LArray<double> weights_;
  
public:
  CalcWeightedFishes(DMatrix<double>& fishes, DArray<double>& weights)
      : fishes_(fishes),
        weights_(weights) {
    this->addArgument(&fishes_);
    this->addArgument(&weights_);
  }

  MSL_USERFUNC
  double operator()(int i, int j, double x) const {
    return fishes_.get(i, j) * weights_.get(i);
  }

};

class CalcBarycenter : public Functor<double, double> {
private:
  double sum_weight_;
public:
  CalcBarycenter()
      : sum_weight_(0) {
  }

  void setSumWeight(double sum_weight) {
    sum_weight_ = sum_weight;
  }

  MSL_USERFUNC
  double operator()(double x) const {
    if (sum_weight_ < 1e-20) {
      return 0;
    }
    return x / sum_weight_;
  }
};

class CalcEuclideanDistance : public msl::Functor2<int, double, double> {
private:
   LMatrix<double> fishes_;
   LArray<double> barycenter_;
   size_t dimensions_;
public:
  CalcEuclideanDistance(DMatrix<double>& fishes, DArray<double>& barycenter,
                        size_t dimensions)
      : fishes_(fishes),
        barycenter_(barycenter, Distribution::COPY),
        dimensions_(dimensions) {
    this->addArgument(&fishes_);
    this->addArgument(&barycenter_);
  }

  MSL_USERFUNC
  double operator()(int i, double x) const {
    double result = 0;
    for (size_t j = 0; j < dimensions_; ++j) {
      result += (fishes_.get(i, j) - barycenter_.get(j))
          * (fishes_.get(i, j) - barycenter_.get(j));
    }
    return sqrt(result);
  }
};

class VolitiveMovement : public Functor3<int, int, double, double> {
private:
  LArray<double> barycenter_;
  LArray<double> distances_;
  double value_min_, value_max_, sum_weight_, sum_weight_last_iteration_,
      step_size_initial_, step_size_final_, step_size_;
  size_t iteration_current_, iteration_max_;
  msl::Rng<double> rng_;
public:
  VolitiveMovement(DArray<double>& barycenter, DArray<double>& distances,
                   double value_min, double value_max, double step_size_initial,
                   double step_size_final, size_t iteration_max,
                   size_t number_of_fishes)
      : barycenter_ { barycenter, Distribution::COPY },
        distances_ { distances },
        value_min_ { value_min },
        value_max_ { value_max },
        sum_weight_ { static_cast<double>(number_of_fishes) },
        sum_weight_last_iteration_ { 0 },
        step_size_initial_ { step_size_initial },
        step_size_final_ { step_size_final },
        step_size_ { step_size_initial },
        iteration_current_ { 0 },
        iteration_max_ { iteration_max },
        rng_ { 0, 1 } {
    this->addArgument(&barycenter_);
    this->addArgument(&distances_);
  }

  void nextIteration(double sum_weight) {
    ++iteration_current_;
    sum_weight_last_iteration_ = sum_weight_;
    sum_weight_ = sum_weight;

    if (iteration_current_ > 1) {
      step_size_ = (step_size_
          - ((step_size_initial_ - step_size_final_)
              / static_cast<double>(iteration_max_ - 1)));
    }
  }

  MSL_USERFUNC
  double operator()(int i, int j, double x) const {
    auto distance = distances_.get(i);
    if (distance < 1e-20) {
      return x;
    }

    double rand_factor = rng_();

    double direction = rand_factor * step_size_ * (value_max_ - value_min_)
        * ((x - barycenter_.get(j)) / distance);

    double result = x;

    if (sum_weight_ > sum_weight_last_iteration_) {
      result -= direction;
    } else {
      result += direction;
    }

    if (result < value_min_) {
      result = value_min_;
    } else if (result > value_max_) {
      result = value_max_;
    }

    return result;
  }
};

////////////////////////////////////////////////////////////////////////
//// Main FSS function
////////////////////////////////////////////////////////////////////////

double fss(size_t number_of_fishes, size_t dimensions, size_t iterations,
           FitnessFunctions fitness_function, bool output, bool progress) {

  // set up config
  auto *conf = &Config::getInstance();
  conf->init(fitness_function, dimensions);

  // create data structures and variables
  DMatrix<double> fishes(number_of_fishes, dimensions, Muesli::num_local_procs,
                         1, Distribution::DIST);

  DMatrix<double> best_fishes(number_of_fishes, dimensions,
                              Muesli::num_local_procs, 1, Distribution::DIST);

  DArray<double> best_fitness(number_of_fishes,
                              std::numeric_limits<double>::lowest(),
                              Distribution::DIST);

  DMatrix<double> candidate_fishes(number_of_fishes, dimensions,
                                   Muesli::num_local_procs, 1,
                                   Distribution::DIST);

  DMatrix<double> fishes_last_iteration(number_of_fishes, dimensions,
                                        Muesli::num_local_procs, 1,
                                        Distribution::DIST);

  DMatrix<double> displacement(number_of_fishes, dimensions,
                               Muesli::num_local_procs, 1, Distribution::DIST);

  DArray<double> fitness(number_of_fishes,
                         std::numeric_limits<double>::lowest(),
                         Distribution::DIST);

  DArray<double> candidate_fitness(number_of_fishes,
                                   std::numeric_limits<double>::lowest(),
                                   Distribution::DIST);

  DArray<double> fitness_last_iteration(number_of_fishes,
                                        std::numeric_limits<double>::lowest(),
                                        Distribution::DIST);

  DArray<double> fitness_variation(number_of_fishes,
                                   std::numeric_limits<double>::lowest(),
                                   Distribution::DIST);

  DArray<double> weight(number_of_fishes, conf->getWeightLowerBound(),
                        Distribution::DIST);

  DArray<double> instinctive_movement_vector(dimensions, Distribution::DIST);

  DArray<double> instinctive_movement_vector_copy_dist(dimensions,
                                                       Distribution::COPY);

  DMatrix<double> weighted_fishes(number_of_fishes, dimensions,
                                  Muesli::num_local_procs, 1,
                                  Distribution::DIST);

  DArray<double> barycenter(dimensions, Distribution::DIST);

  DArray<double> barycenter_copy_dist(dimensions, Distribution::COPY);

  DArray<double> distances_fishes_barycenter(number_of_fishes,
                                             Distribution::DIST);

  // intantiate functors
  InitFishes init_functor { conf->getInitLowerBound(), conf->getInitUpperBound() };
  CopyFishes copy_fishes_functor { fishes };
  CalcFitness calc_fitness_functor = CalcFitness { fishes, dimensions,
      fitness_function };
  CalcFitness calc_fitness_candidates_functor = CalcFitness { candidate_fishes,
      dimensions, fitness_function };
  IndividualMovement individual_movement_functor { fishes,
      conf->getLowerBound(), conf->getUpperBound(), conf->getStepSizeInitial(),
      conf->getStepSizeFinal(), iterations };
  CopyCandidates copy_candidates_functor { candidate_fishes, candidate_fitness,
      fitness };

  CopyCandidates copy_best_fishes_functor { fishes, fitness, best_fitness };
  UpdateFitness update_best_fitness_functor { fitness };

  UpdateFitness update_fitness_functor { candidate_fitness };
  CopyFitness copy_fitness_functor { fitness };
  CalcFitnessVariation calc_fitness_variation_functor { fitness,
      fitness_last_iteration };
  MaxArray max_fitness_variation_functor { };
  Feeding feeding_functor { fitness_variation, conf->getWeightLowerBound(), conf
      ->getWeightUpperBound() };
  CalcInstinctiveMovementVector calc_instinctive_movement_vector { };
  SumArray sum_array_functor { };
  SumMatrix sum_matrix_functor { };

  MZipDif dif_zip_matrix_functor { };
  CalcDisplacement calc_displacement_functor { fishes, fishes_last_iteration };

  CalcDisplacementFitnessVector calc_displacement_fitness_vector {
      fitness_variation };
  InstinctiveMovement instinctive_movement_functor {
      instinctive_movement_vector_copy_dist, conf->getLowerBound(), conf
          ->getInitUpperBound() };
  CalcWeightedFishes calc_weighted_fishes { fishes, weight };
  CalcBarycenter calc_barycenter_functor { };
  CalcEuclideanDistance calc_euclidean_distance_functor { fishes,
      barycenter_copy_dist, dimensions };
  VolitiveMovement volitive_movement_functor { barycenter_copy_dist,
      distances_fishes_barycenter, conf->getLowerBound(), conf->getUpperBound(),
      conf->getStepSizeVolitiveInitial(), conf->getStepSizeVolitiveFinal(),
      iterations, number_of_fishes };

  // init fish matrix
  fishes.mapInPlace(init_functor);

  // start iterations
  for (size_t current_iteration = 0; current_iteration < iterations;
      ++current_iteration) {

    // calculate fitness
    fitness.mapIndexInPlace(calc_fitness_functor);

    // store best results
    best_fishes.mapIndexInPlace(copy_best_fishes_functor);
    best_fitness.mapIndexInPlace(update_best_fitness_functor);

    fishes_last_iteration.mapIndexInPlace(copy_fishes_functor);
    fitness_last_iteration.mapIndexInPlace(copy_fitness_functor);

/////////////////////////////////////////////////////////////////////////////////
// individual movement
    individual_movement_functor.nextIteration();
    candidate_fishes.mapIndexInPlace(individual_movement_functor);

    candidate_fitness.mapIndexInPlace(calc_fitness_candidates_functor);

    fishes.mapIndexInPlace(copy_candidates_functor);
    fitness.mapIndexInPlace(update_fitness_functor);

    // store best results
    best_fishes.mapIndexInPlace(copy_best_fishes_functor);
    best_fitness.mapIndexInPlace(update_best_fitness_functor);

/////////////////////////////////////////////////////////////////////////////////
// feeding; required for volitive movement
    fitness_variation.mapIndexInPlace(calc_fitness_variation_functor);

    double max_fitness_variation = fitness_variation.fold(
        max_fitness_variation_functor, true);

    feeding_functor.setMaxFitnessVariation(max_fitness_variation);
    weight.mapIndexInPlace(feeding_functor);

///////////////////////////////////////////////////////////////////////////////////
    // instinctive movement
    double sum_fitness_variation = fitness_variation.fold(sum_array_functor,
                                                          true);
    displacement.mapIndexInPlace(calc_displacement_functor);

    displacement.mapIndexInPlace(calc_displacement_fitness_vector);  // delta x * delta f

    instinctive_movement_vector = displacement.foldCols(sum_matrix_functor);  // sum displacement * delta f

    calc_instinctive_movement_vector.setSumFitnessVariation(
        sum_fitness_variation);

    instinctive_movement_vector.mapInPlace(calc_instinctive_movement_vector);  // divide sum by (sum fitness variation)
    instinctive_movement_vector.gather(instinctive_movement_vector_copy_dist);

    fishes.mapIndexInPlace(instinctive_movement_functor);  // apply vector

    ///////////////////////////////////////////////////////////////////////////////
    // volitive movement

    double sum_weight = weight.fold(sum_array_functor, true);
    weighted_fishes.mapIndexInPlace(calc_weighted_fishes);

    barycenter = weighted_fishes.foldCols(sum_matrix_functor);

    calc_barycenter_functor.setSumWeight(sum_weight);
    barycenter.mapInPlace(calc_barycenter_functor);
    barycenter.gather(barycenter_copy_dist);

    distances_fishes_barycenter.mapIndexInPlace(
        calc_euclidean_distance_functor);

    volitive_movement_functor.nextIteration(sum_weight);
    fishes.mapIndexInPlace(volitive_movement_functor);

    if (progress) {
      int barWidth = 70;
      if (current_iteration == iterations - 1) {
        for (int i = 0; i < barWidth + 10; ++i) {
          msl::printv(" ");
        }
        msl::printv("\r");
      } else {
        double progress = static_cast<double>(current_iteration)
            / static_cast<double>(iterations);

        msl::printv("[");
        int pos = barWidth * progress;

        for (int i = 0; i < barWidth; ++i) {
          if (i < pos)
            msl::printv("=");
          else if (i == pos)
            msl::printv(">");
          else
            msl::printv(" ");
        }
        msl::printv("] %i%%\r", static_cast<int>(progress * 100.0));
      }
      fflush(stdout);
    }

  }  // end of fss iteration loop

  ////////////////////////////////////////////////////////////////////////////////////////////////////
//  collect final result

  // local final results
  size_t local_procs = static_cast<size_t>(Muesli::num_local_procs);
  double local_best_fitness = std::numeric_limits<double>::lowest();
  size_t local_best_fitness_index = 0;
  size_t local_size = fishes.getLocalRows();

  best_fitness.download();
  double* local_fitness_array = best_fitness.getLocalPartition();

  for (size_t i = 0; i < local_size; ++i) {
    double fitness_candidate = local_fitness_array[i];
    if (fitness_candidate > local_best_fitness) {
      local_best_fitness_index = i;
      local_best_fitness = fitness_candidate;
    }
  }

  // set results
  DMatrix<double> local_best_fishes(Muesli::num_local_procs, dimensions,
                                    Muesli::num_local_procs, 1,
                                    Distribution::DIST);

  DArray<double> local_best_fitnesses(Muesli::num_local_procs,
                                      std::numeric_limits<double>::lowest(),
                                      Distribution::DIST);

  local_best_fitnesses.set(Muesli::proc_id, local_best_fitness);

  best_fishes.download();
  for (size_t i = 0; i < dimensions; ++i) {

    local_best_fishes.set(Muesli::proc_id, i,
                          best_fishes.getLocal(local_best_fitness_index, i));
  }

  // global final result
  double global_best_fitness = std::numeric_limits<double>::lowest();
  size_t global_best_fitness_index = 0;

// get index and fitness of best solution
  double* global_fitness_array = new double[Muesli::num_local_procs];

  local_best_fitnesses.gather(global_fitness_array);

  for (size_t i = 0; i < local_procs; ++i) {
    double fitness_candidate = global_fitness_array[i];
    if (fitness_candidate > global_best_fitness) {
      global_best_fitness_index = i;
      global_best_fitness = fitness_candidate;
    }
  }

  if (output) {
    local_best_fishes.broadcastPartition(global_best_fitness_index, 0);

    msl::printv("Fish: [");
    for (size_t j = 0; j < dimensions - 1; ++j) {
      msl::printv("%e, ", local_best_fishes.getLocal(0, j));
    }
    msl::printv("%e]\n", local_best_fishes.getLocal(0, dimensions - 1));
  }

  msl::printv("Fitness: %e\n", global_best_fitness);
  msl::printv("Weight: %e\n\n", weight.get(global_best_fitness_index));

  delete[] global_fitness_array;

  return global_best_fitness;
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
