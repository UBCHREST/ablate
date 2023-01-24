#ifndef ABLATELIBRARY_REVERSERANGE_HPP
#define ABLATELIBRARY_REVERSERANGE_HPP
#include <vector>
#include "petsc.h"
#include "range.hpp"
namespace ablate::solver {

/**
 * Results in a mapping from the point (face/cell id) to the index in the range
 */
struct ReverseRange {
   private:
    PetscInt rangeStart = 0;
    PetscInt pointStart = 0;
    std::vector<PetscInt> indices;

   public:
    explicit ReverseRange(const Range& range) {
        rangeStart = range.start;
        if (range.points && (range.start != range.end)) {
            // find the min/max point
            PetscInt maxPoint = rangeStart;
            for (PetscInt i = range.start; i < range.end; ++i) {
                pointStart = PetscMin(pointStart, range.GetPoint(i));
                maxPoint = PetscMax(maxPoint, range.GetPoint(i));
            }

            // size to the maximum location, set default to -1
            indices.resize(maxPoint - pointStart + 1, -1);

            // store the index at each point
            for (PetscInt i = range.start; i < range.end; ++i) {
                indices[range.GetPoint(i) - pointStart] = i;
            }
        }
    }

    ReverseRange() : rangeStart(-1), pointStart(-1) {}

    /**
     * Get the index for point i
     * @param point
     * @return
     */
    inline PetscInt GetIndex(PetscInt point) const { return indices.empty() ? point : indices[point - pointStart]; }

    /**
     * Get the absolute index for this point.  This always starts at zero
     * @param point
     * @return
     */
    inline PetscInt GetAbsoluteIndex(PetscInt point) const { return GetIndex(point) - rangeStart; }
};
}  // namespace ablate::solver
#endif  // ABLATELIBRARY_RANGE_HPP
