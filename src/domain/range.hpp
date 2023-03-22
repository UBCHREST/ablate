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
    const PetscInt *points = nullptr;

    /**
     * Get the point for the index i
     * @param i
     * @return
     */
    inline PetscInt GetPoint(PetscInt i) const { return points ? points[i] : i; }
};

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
    inline const ablate::domain::Range &GetRange() {
        range.points = points.data();
        range.start = 0;
        range.end = (PetscInt)points.size();
        return range;
    }
};

/**
 * Results in a mapping from the point (face/cell id) to the index in the range
 */
struct ReverseRange {
   private:
    PetscInt rangeStart = 0;
    PetscInt pointStart = 0;
    std::vector<PetscInt> indices;

   public:
    explicit ReverseRange(const ablate::domain::Range &range) {
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

void GetRange(DM dm, const std::shared_ptr<Region> region, PetscInt depth, Range &range);
void GetCellRange(DM dm, const std::shared_ptr<Region> region, Range &cellRange);
void GetFaceRange(DM dm, const std::shared_ptr<Region> region, Range &faceRange);
void RestoreRange(Range &range);

}  // namespace ablate::domain
#endif  // ABLATELIBRARY_RANGE_HPP
