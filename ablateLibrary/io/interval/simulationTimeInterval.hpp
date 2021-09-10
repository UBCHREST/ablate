#ifndef ABLATELIBRARY_SIMULATIONTIMEINTERVAL_HPP
#define ABLATELIBRARY_SIMULATIONTIMEINTERVAL_HPP

#include "interval.hpp"
namespace ablate::io::interval {

class SimulationTimeInterval : public Interval {
   private:
    const double timeInterval;
    double nextTime;

   public:
    explicit SimulationTimeInterval(double timeInterval);
    bool Check(MPI_Comm comm, PetscInt steps, PetscReal time) override;
};

}  // namespace ablate::io::interval

#endif  // ABLATELIBRARY_SIMULATIONTIMEINTERVAL_HPP
