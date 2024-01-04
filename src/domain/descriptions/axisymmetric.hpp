#ifndef ABLATELIBRARY_AXISYMMETRIC_HPP
#define ABLATELIBRARY_AXISYMMETRIC_HPP

#include <array>
#include <memory>
#include <vector>
#include "meshDescription.hpp"

namespace ablate::domain::descriptions {

/**
 * produces an axisymmetric mesh around the z axis
 */
class Axisymmetric : public ablate::domain::descriptions::MeshDescription {
   private:
    //! Store the start and end location of the mesh
    const std::array<PetscReal, 3> startLocation;
    const PetscReal length;//this is in z

    //! hard code the assumed mesh dimension
    inline static const PetscInt dim = 3;

    //! Store the number of wedges, wedges/pie slices in the circle
    const PetscInt numberWedges;

    //! Store the number of slices, slicing of the cylinder along the z axis
    const PetscInt numberSlices;

    //! Store the number of cells per slice
    const PetscInt numberCellsPerSlice;

    //! Store the number of vertices per half slice
    const PetscInt numberVerticesPerHalfSlice;

    //! Compute the number of cells
    const PetscInt numberCells;

    //! And the number of vertices
    const PetscInt numberVertices;

   public:
    /**
     * generate and precompute a bunch of the required parameters
     * @param startLocation
     * @param endLocation
     * @param numberWedges
     * @param numberSlices
     */
    Axisymmetric(std::vector<PetscReal> startLocation, PetscReal length, PetscInt numberWedges, PetscInt numberSlices);

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
    [[nodiscard]] DMPolytopeType GetCellType(PetscInt cell) const override { return DM_POLYTOPE_TRI_PRISM; }

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

};  // namespace ablate::domain
}  // namespace ablate::domain::descriptions
#endif  // ABLATELIBRARY_AXISYMMETRIC_HPP
