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
    //! label to create
    const std::vector<std::shared_ptr<domain::Region>> regions;

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
    explicit ExtrudeLabel(std::vector<std::shared_ptr<domain::Region>>, double thickness = {});

    void Modify(DM &) override;

    [[nodiscard]] std::string ToString() const override;
};

}  // namespace ablate::domain::modifiers

#endif  // ABLATELIBRARY_EXTRUDELABEL_HPP
