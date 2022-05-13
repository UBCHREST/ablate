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
class ExtrudeLabel  : public Modifier {
   private:
    //! label to create
    const std::vector<std::shared_ptr<domain::Region>> regions;

    /**
     * ABLATE implementation of DMPlexTransformAdaptLabel that uses a petsc options object to setup the transform
     * @param dm
     * @param metric
     * @param adaptLabel
     * @param rgLabel
     * @param rdm
     * @return
     */
    PetscErrorCode DMPlexTransformAdaptLabel(DM dm, PETSC_UNUSED Vec metric, DMLabel adaptLabel, PETSC_UNUSED DMLabel rgLabel, DM *rdm);

   public:
    explicit ExtrudeLabel(std::vector<std::shared_ptr<domain::Region>>);

    void Modify(DM&) override;

    std::string ToString() const override;

};

}

#endif  // ABLATELIBRARY_EXTRUDELABEL_HPP
