#ifndef ABLATELIBRARY_FIXEDINTERVAL_HPP
#define ABLATELIBRARY_FIXEDINTERVAL_HPP

#include "interval.hpp"
namespace ablate::io::interval {

/**
 * Simple fixed interval that outputs every n steps
 */
class FixedInterval : public Interval {
   private:
    const int interval;

   public:
    explicit FixedInterval(int interval = {});
    bool Check(MPI_Comm comm, PetscInt steps, PetscReal time) override;
};

}  // namespace ablate::io::interval
#endif  // ABLATELIBRARY_FIXEDINTERVAL_HPP
