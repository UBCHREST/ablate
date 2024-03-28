#ifndef ABLATELIBRARY_AXISYMMETRIC_HPP
#define ABLATELIBRARY_AXISYMMETRIC_HPP

#include <array>
#include <memory>
#include <vector>
#include "axisDescription.hpp"
#include "mathFunctions/mathFunction.hpp"
#include "meshDescription.hpp"

namespace ablate::domain::descriptions {

/**
 * produces an axisymmetric mesh around the z axis
 */
class Axisymmetric : public ablate::domain::descriptions::MeshDescription {
   private:
    //! Store the start, end, and nodes in the axis of the mesh
    const std::shared_ptr<ablate::domain::descriptions::AxisDescription> axisDescription;

    //! function used to describe a single return value (radius) as a functino of z
    const std::shared_ptr<ablate::mathFunctions::MathFunction> radiusFunction;

    //! hard code the assumed mesh dimension
    inline static const PetscInt dim = 3;

    //! Store the number of wedges, wedges/pie slices in the circle
    const PetscInt numberWedges;

    //! Store the number of slices, slicing of the cylinder along the z axis
    const PetscInt numberSlices;

    //! Store the number of shells, slicing of the cylinder along the radius
    const PetscInt numberShells;

    //! Store the number of cells per slice
    const PetscInt numberCellsPerSlice;

    //! store the number of cells per slice
    const PetscInt numberCellsPerShell;

    //! Store the number of vertices per shell, does not include the center
    const PetscInt numberVerticesPerShell;

    //! Store the number of vertices in the center
    const PetscInt numberCenterVertices;

    //! Compute the number of cells
    const PetscInt numberCells;

    //! And the number of vertices
    const PetscInt numberVertices;

    //! store the number of tri prism cells to simplify the logic
    const PetscInt numberTriPrismCells;

    //! function used to describe a different boundary regions on the outer shell boundary
    const std::shared_ptr<ablate::mathFunctions::MathFunction> boundaryFunction;

    //! precompute the region identifier for the boundary
    const static inline std::shared_ptr<ablate::domain::Region> shellBoundary = std::make_shared<ablate::domain::Region>("outerShell");
    const static inline std::shared_ptr<ablate::domain::Region> lowerCapBoundary = std::make_shared<ablate::domain::Region>("lowerCap");
    const static inline std::shared_ptr<ablate::domain::Region> upperCapBoundary = std::make_shared<ablate::domain::Region>("upperCap");

    /**
     * Reverse the cell order to make sure hexes appear before tri prism
     * @param in
     * @return
     */
    [[nodiscard]] inline PetscInt CellReverser(PetscInt in) const { return numberCells - in - 1; }

   public:
    /**
     * generate and precompute a bunch of the required parameters
     * @param axis describes the mesh along the z axis, must be 3D
     * @param radiusFunction a radius function that describes the radius as a function of z
     * @param numberWedges wedges/pie slices in the circle
     * @param numberShells slicing of the cylinder along the radius
     */
    Axisymmetric(std::shared_ptr<ablate::domain::descriptions::AxisDescription> axis, std::shared_ptr<ablate::mathFunctions::MathFunction> radiusFunction, PetscInt numberWedges, PetscInt numberShells,
                 std::shared_ptr<ablate::mathFunctions::MathFunction> boundaryFunction);

    /**
     * The overall assumed dimension of the mesh
     * @return
     */
    [[nodiscard]] const PetscInt& GetMeshDimension() const override { return dim; }

    /**
     * Add some getters, so we can virtualize this in the future if needed
     * @return
     */
    [[nodiscard]] const PetscInt& GetNumberCells() const override { return numberCells; }

    /**
     * Add some getters, so we can virtualize this in the future if needed
     * @return
     */
    [[nodiscard]] const PetscInt& GetNumberVertices() const override { return numberVertices; }

    /**
     * Return the cell type for this cell, based off zero offset
     * @return
     */
    [[nodiscard]] DMPolytopeType GetCellType(PetscInt cell) const override;

    /**
     * Builds the topology based upon a zero vertex offset.  The cellNodes should be sized to hold cell
     * @return
     */
    void BuildTopology(PetscInt cell, PetscInt* cellNodes) const override;

    /**
     * Builds the topology based upon a zero vertex offset.  The cellNodes should be sized to hold cell
     * @return
     */
    void SetCoordinate(PetscInt node, PetscReal* coordinate) const override;

    /**
     * Returns a hard coded boundary region name
     * @return
     */
    [[nodiscard]] std::shared_ptr<ablate::domain::Region> GetBoundaryRegion() const override { return std::make_shared<ablate::domain::Region>("boundary"); }

    /**
     * returns the boundary region for the end caps and outer shell
     * @param face
     * @return
     */
    [[nodiscard]] std::shared_ptr<ablate::domain::Region> GetRegion(const std::set<PetscInt>& face) const override;
};
}  // namespace ablate::domain::descriptions
#endif  // ABLATELIBRARY_AXISYMMETRIC_HPP
