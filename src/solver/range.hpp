#ifndef ABLATELIBRARY_RANGE_HPP
#define ABLATELIBRARY_RANGE_HPP
#include "petsc.h"

namespace ablate::solver {

/**
 * Simple struct used to describe a range in the dm
 */
struct Range {
    IS is = nullptr;
    PetscInt start;
    PetscInt end;
    const PetscInt* points = nullptr;
};
}  // namespace ablate::solver
#endif  // ABLATELIBRARY_RANGE_HPP
