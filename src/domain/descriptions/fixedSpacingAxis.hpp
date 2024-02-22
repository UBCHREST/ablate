#ifndef ABLATELIBRARY_FIXEDSPACINGAXIS_HPP
#define ABLATELIBRARY_FIXEDSPACINGAXIS_HPP

#include <array>
#include <memory>
#include <vector>
#include "axisDescription.hpp"
#include "mathFunctions/mathFunction.hpp"

namespace ablate::domain::descriptions {

/**
 * Describes a simple strait line along the z axis
 */
class FixedSpacingAxis : public ablate::domain::descriptions::AxisDescription {
   private:
    //! Store the start and end location of the mesh
    const std::array<PetscReal, 3> startLocation;

    //! the length of the domain
    const PetscReal length;  // this is in z

    //! the number of nodes in the axis
    const PetscInt numberNodes;

   public:
    /**
     * generate and precompute a bunch of the required parameters
     * @param startLocation the start coordinate of the mesh, must be 3D
     * @param length the length of the domain starting at the start coordinate
     * @param numberNodes the number of nodes along the axis
     */
    FixedSpacingAxis(const std::vector<PetscReal>& startLocation, PetscReal length, PetscInt numberNodes);

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
#endif  // ABLATELIBRARY_AXISYMMETRIC_HPP
