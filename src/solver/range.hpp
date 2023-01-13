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

    /**
     * Get the point for the index i
     * @param i
     * @return
     */
    inline PetscInt GetPoint(PetscInt i) const { return points ? points[i] : i; }
};
}  // namespace ablate::solver
#endif  // ABLATELIBRARY_RANGE_HPP
