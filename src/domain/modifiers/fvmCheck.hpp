#ifndef ABLATELIBRARY_FVMCHECK_HPP
#define ABLATELIBRARY_FVMCHECK_HPP

#include <domain/region.hpp>
#include <memory>
#include "mathFunctions/mathFunction.hpp"
#include "modifier.hpp"
#include "solver/range.hpp"

namespace ablate::domain::modifiers {

/**
 * The FVM check marches over each face in a mesh region, sums the contributions for each cell in the region to ensure they sum to 0.0
 */
class FvmCheck : public Modifier {
   private:
    //! the fvm region to check
    const std::shared_ptr<domain::Region> region;

    //! The number of expected faces for each cell
    const PetscInt expectedFaceCount;

    //! The number of expected nodes for each cell
    const PetscInt expectedNodeCount;

   private:
    /**
     * modified version of the get range call
     * @param dm
     * @param depth
     * @param range
     */
    void GetRange(DM dm, PetscInt depth, ablate::solver::Range &range) const;

    /**
     * modified version of the restore range call
     */
    static void RestoreRange(DM dm, ablate::solver::Range &range);

   public:
    /**
     * The region to check over the boundary cells
     * @param regions
     */
    explicit FvmCheck(std::shared_ptr<domain::Region> fvmRegion, int expectedFaceCount = {}, int expectedNodeCount = {});

    /**
     * Check the supplied dm
     */
    void Modify(DM &) override;

    /**
     * Get the id for the fvm check
     * @return
     */
    [[nodiscard]] std::string ToString() const override;
};

}  // namespace ablate::domain::modifiers
#endif  // ABLATELIBRARY_FVMCHECK_HPP
