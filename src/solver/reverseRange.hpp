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
    PetscInt start;
    std::vector<PetscInt> indices;

   public:
    explicit ReverseRange(const Range& range) {
        if (range.points) {
            for (PetscInt i = range.start; i < range.end; ++i) {
                indices.push_back(range.GetPoint(i));
            }
        } else {
            start = range.start;
        }
    }

    ReverseRange(): start(-1) {
    }

    /**
     * Get the point for the index i
     * @param i
     * @return
     */
    inline PetscInt GetIndex(PetscInt point) const { return indices.empty() ? start + point : indices[point]; }
};
}  // namespace ablate::solver
#endif  // ABLATELIBRARY_RANGE_HPP
