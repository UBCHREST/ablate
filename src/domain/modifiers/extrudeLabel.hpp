#ifndef ABLATELIBRARY_EXTRUDELABEL_HPP
#define ABLATELIBRARY_EXTRUDELABEL_HPP

#include <domain/region.hpp>
#include <memory>
#include "mathFunctions/mathFunction.hpp"
#include "modifier.hpp"

namespace ablate::domain::modifiers {

/**
 * Extrude faces outward of the domain for the specified labels
 */
class ExtrudeLabel : public Modifier {
   private:
    //! boundary labels to create.  The labels must be complete
    const std::vector<std::shared_ptr<domain::Region>> regions;

    //! optional label to tag the new boundary interface (between original and extruded regions)
    const std::shared_ptr<domain::Region> boundaryRegion;

    //! optional label to tag the original cell
    const std::shared_ptr<domain::Region> originalRegion;

    //! optional label to tag the newly extruded cell
    std::shared_ptr<domain::Region> extrudedRegion;

    //! thickness for the extruded cells. If default (0) the 2 * minimum cell radius is used
    const double thickness;

    /**
     * ABLATE implementation of DMPlexTransformAdaptLabel that uses a petsc options object to setup the transform
     * @param dm
     * @param metric
     * @param adaptLabel
     * @param rgLabel
     * @param transformOptions, the options for the new transform.  nullptr is global options
     * @param rdm
     * @return
     */
    static PetscErrorCode DMPlexTransformAdaptLabel(DM dm, PETSC_UNUSED Vec metric, DMLabel adaptLabel, PETSC_UNUSED DMLabel rgLabel, PetscOptions transformOptions, DM *rdm);

   public:
    /**
     * extrude a label on a boundary
     * @param regions the label(s) describing the faces to extrude
     * @param boundaryRegion the new label describing the faces between the original and extruded regions
     * @param originalRegion label describing the original mesh
     * @param extrudedRegion label describing the new extruded cells
     * @param thickness
     */
    explicit ExtrudeLabel(std::vector<std::shared_ptr<domain::Region>> regions, std::shared_ptr<domain::Region> boundaryRegion, std::shared_ptr<domain::Region> originalRegion,
                          std::shared_ptr<domain::Region> extrudedRegion, double thickness = {});

    void Modify(DM &) override;

    [[nodiscard]] std::string ToString() const override;
};

}  // namespace ablate::domain::modifiers

#endif  // ABLATELIBRARY_EXTRUDELABEL_HPP
