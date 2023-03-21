#ifndef ABLATELIBRARY_RANGE_HPP
#define ABLATELIBRARY_RANGE_HPP
#include "petsc.h"
#include "region.hpp"

namespace ablate::domain {

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




void GetRange(DM dm, const std::shared_ptr<Region> region, PetscInt depth, Range &range);
void GetCellRange(DM dm, const std::shared_ptr<Region> region, Range &cellRange);
void GetFaceRange(DM dm, const std::shared_ptr<Region> region, Range &faceRange);
void RestoreRange(Range &range);


}  // namespace ablate::domain
#endif  // ABLATELIBRARY_RANGE_HPP
