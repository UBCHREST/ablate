#ifndef ABLATELIBRARY_INTERVAL_HPP
#define ABLATELIBRARY_INTERVAL_HPP

#include <petsc.h>

namespace ablate::io::interval {

/**
 * Simple interface to determine if an output should be performed
 */
class Interval {
   public:
    virtual ~Interval() = default;

    virtual bool Check(MPI_Comm comm, PetscInt steps, PetscReal time) = 0;
};
}  // namespace ablate::io::interval

#endif  // ABLATELIBRARY_INTERVAL_HPP
