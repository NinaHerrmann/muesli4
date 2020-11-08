#pragma once

#include "muesli.h"
#include "dmatrix.h"
#include "darray.h"
#include "lmatrix.h"
#include "larray.h"
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

class CalcFitness : public msl::AMapIndexFunctor<double, double> {
 public:
  CalcFitness(DMatrix<double>& fishes, size_t dimensions, FitnessFunctions fitness_function);

  MSL_USERFUNC
  double operator()(int i, double x) const;

  MSL_USERFUNC
  double ackley(int i, double x) const;

  MSL_USERFUNC
  double griewank(int i, double x) const;

  MSL_USERFUNC
  double rastrigin(int i, double x) const;

  MSL_USERFUNC
  double rosenbrock(int i, double x) const;

  MSL_USERFUNC
  double schwefel(int i, double x) const;

  MSL_USERFUNC
  double sphere(int i, double x) const;

 protected:
  LMatrix<double> fishes_;
  size_t dimensions_;
  FitnessFunctions fitness_function_;
};


//////////////////////////////////////////////////////////////////////////
//// config
//////////////////////////////////////////////////////////////////////////

class Config {
 public:

  ~Config();

  static Config& getInstance() {
    static Config instance;
    return instance;
  }

  void init(FitnessFunctions ff, size_t problem_dimensions);

  Config(Config const&) = delete;
  void operator=(Config const&) = delete;

  size_t getProblemDimensions() const;

  double getUpperBound() const;
  double getLowerBound() const;
  double getProblemRange() const;
  double getInitUpperBound() const;
  double getInitLowerBound() const;
  double getWeightUpperBound() const;
  double getWeightLowerBound() const;

  double getStepSizeInitial() const;
  double getStepSizeFinal() const;
  double getStepSizeVolitiveInitial() const;
  double getStepSizeVolitiveFinal() const;

//  FitnessFunction* getFitnessFunctor() const;

 private:
  Config() {
  }

  size_t problem_dimensions_;
  double upper_bound_;
};

}
}
