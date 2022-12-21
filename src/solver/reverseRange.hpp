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
    PetscInt indexStart = 0;
    std::vector<PetscInt> indices;

   public:
    explicit ReverseRange(const Range& range) {
        rangeStart = range.start;
        if (range.points) {
            // size to the maximum location, set default to -1
            indexStart = range.GetPoint(range.start);
            indices.resize(range.GetPoint(range.end - 1) - indexStart, -1);

            // store the index at each point
            for (PetscInt i = range.start; i < range.end; ++i) {
                indices[range.GetPoint(i) - indexStart] = i;
            }
        }
    }

    ReverseRange() : rangeStart(-1), indexStart(-1) {}

    /**
     * Get the index for point i
     * @param point
     * @return
     */
    inline PetscInt GetIndex(PetscInt point) const { return indices.empty() ? point : indices[point - indexStart]; }

    /**
     * Get the absolute index for this point.  This always starts at zero
     * @param point
     * @return
     */
    inline PetscInt GetAbsoluteIndex(PetscInt point) const { return GetIndex(point) - rangeStart; }
};
}  // namespace ablate::solver
#endif  // ABLATELIBRARY_RANGE_HPP
