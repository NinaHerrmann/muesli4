
////////////////////////////////////////////////////////////////////////
//// FSS functors
////////////////////////////////////////////////////////////////////////

namespace msl {
namespace fss {

class InitFishes : public Functor3<int, int, double, double> {
private:
  double min;
  double max;

public:
  InitFishes(double mn, double mx) : min(mn), max(mx){}

  MSL_USERFUNC
  double operator()(int i, int j, double x) const {
    return ((myRand(i*100000+j) / ((double) 2.0 * maxRand) * (max-min) + min;
  }
  
  MSL_USERFUNC
  int myRand(int state) const {
    return (multiplier * state + increment) % maxRand;
  }
};

// *********************************************************************************
class CopyFishes : public Functor<double, double> {
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

// *********************************************************************************
class IndividualMovement : public Functor<double, double> {
private:
  double value_min_, 
  double value_max_, 
  double step_size_initial_, 
  double step_size_final_, 
  double step_size_;
  size_t iteration_max_;
  msl::Rng<double> rng_;
public:
  IndividualMovement(DMatrix<double>& fishes, double value_min,
                     double value_max, double step_size_initial,
                     double step_size_final, size_t iteration_max)
      : fishes_(fishes),
        value_min_ (value_min),
        value_max_ (value_max),
        step_size_initial_ (step_size_initial),
        step_size_final_ (step_size_final),
        step_size_ (step_size_initial),
		iteration_max_ (iteration_max),
        rng_ { -1, 1 } {
    this->addArgument(&fishes_);
  }

  void reduceStepSize() {
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

// *********************************************************************************
class CopyCandidates : public Functor<double, double> {
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

// *********************************************************************************
class UpdateFitness : public Functor<double, double> {
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

// *********************************************************************************
class CopyFitness : public Functor<double, double> {
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

// *********************************************************************************
class CalcFitnessVariation : public Functor<double, double> {
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

// *********************************************************************************
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

// *********************************************************************************
class SumMatrix : public Functor2<double, double> {
public:
  MSL_USERFUNC
  double operator()(double a, double b) const {
    return a + b;
  }
};

// *********************************************************************************
class MZipDif : public Functor2<double, double, double> {
 public:
  MSL_USERFUNC
  double operator()(double a, double b) const {
    return a - b;
  }
};

// *********************************************************************************
class CalcDisplacement : public Functor<double, double> {
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

// *********************************************************************************
class Feeding : public Functor<double, double> {
 private:
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

// *********************************************************************************
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
}

// *********************************************************************************
class CalcInstinctiveMovementVector : public Functor<double, double> {
private:
  double sum_fitness_variation_;
  
public:
  CalcInstinctiveMovementVector(double variation)
      : sum_fitness_variation_(variation) {
  }

  MSL_USERFUNC
  double operator()(double x) const {
    if (sum_fitness_variation_ < 1e-20) {
      return 0;
    }
    return x / sum_fitness_variation_;
  }
}

// *********************************************************************************
class InstinctiveMovement : public Functor2<double, double, double> {
private:
  double value_min_, value_max_;
  
public:
  InstinctiveMovement(double value_min, double value_max): 
        value_min_ (value_min),
        value_max_ (value_max) {}

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
}

// *********************************************************************************
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
}

// *********************************************************************************
class CalcBarycenter : public Functor<double, double> {
private:
  double sum_weight_;
public:
  CalcBarycenter(double weight)
      : sum_weight_(weight) {}

  MSL_USERFUNC
  double operator()(double x) const {
    if (sum_weight_ < 1e-20) {
      return 0;
    }
    return x / sum_weight_;
  }
}

// *********************************************************************************
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
}

// *********************************************************************************
class VolitiveMovement : public Functor3<int, int, double, double> {
private:
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
}
}} // close namespaces