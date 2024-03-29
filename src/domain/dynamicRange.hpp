#ifndef ABLATELIBRARY_DYNAMICRANGE_HPP
#define ABLATELIBRARY_DYNAMICRANGE_HPP
#include <vector>
#include "petsc.h"
#include "range.hpp"

namespace ablate::domain {

/**
 * An helper class to build a range dynamically
 */
class DynamicRange {
   private:
    // The points in this range
    std::vector<PetscInt> points;

    // the updated range object
    ablate::domain::Range range;

   public:
    /**
     * Add point to the range
     * @param p
     */
    inline void Add(PetscInt p) { points.push_back(p); }

    /**
     * Get the current range
     * @return
     */
    inline const ablate::domain::Range& GetRange() {
        range.points = points.data();
        range.start = 0;
        range.end = (PetscInt)points.size();
        return range;
    }
};
}  // namespace ablate::domain
#endif  // ABLATELIBRARY_RANGE_HPP
