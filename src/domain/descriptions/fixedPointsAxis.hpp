#ifndef ABLATELIBRARY_FIXEDPOINTSAXIS_HPP
#define ABLATELIBRARY_FIXEDPOINTSAXIS_HPP

#include <array>
#include <memory>
#include <vector>
#include "axisDescription.hpp"
#include "mathFunctions/mathFunction.hpp"

namespace ablate::domain::descriptions {

/**
 * Describes a simple strait line along the z axis with specified offsets
 */
class FixedPointsAxis : public ablate::domain::descriptions::AxisDescription {
   private:
    //! Store the start and end location of the mesh
    const std::array<PetscReal, 3> startLocation;

    //! Store the z offset from each node
    const std::vector<PetscReal> zOffset;

    //! store the number of vertices for quick lookup
    const PetscInt numberNodes;

   public:
    /**
     * read in and compute the axis from a list of z offsets and optional start location
     * @param nodeOffsets z offsets from the start location
     * @param startLocation optional start location (default is {0, 0, 0})
     */
    explicit FixedPointsAxis(const std::vector<PetscReal>& nodeOffsets, const std::vector<PetscReal>& startLocation = {});

    /**
     * Total number of nodes/vertices in the entire mesh
     * @return
     */
    [[nodiscard]] const PetscInt& GetNumberVertices() const override { return numberNodes; }

    /**
     * Sets the axis coordinate for this node
     * @return
     */
    void SetCoordinate(PetscInt node, PetscReal coordinate[3]) const override;
};
}  // namespace ablate::domain::descriptions
#endif  // ABLATELIBRARY_FIXEDPOINTSAXIS_HPP
