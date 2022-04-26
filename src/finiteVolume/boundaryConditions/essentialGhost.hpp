#ifndef ABLATELIBRARY_ESSENTIALGHOST_HPP
#define ABLATELIBRARY_ESSENTIALGHOST_HPP

#include <mathFunctions/fieldFunction.hpp>
#include "ghost.hpp"

namespace ablate::finiteVolume::boundaryConditions {

class EssentialGhost : public Ghost {
   private:
    static PetscErrorCode EssentialGhostUpdate(PetscReal time, const PetscReal* c, const PetscReal* n, const PetscScalar* a_xI, PetscScalar* a_xG, void* ctx);

    const std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction;

    /**
     * Uses linear interpolation to force the value at the face
     */
    const bool enforceAtFace;

   public:
    EssentialGhost(std::string boundaryName, std::vector<int> labelId, std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction, std::string labelName = {}, bool enforceAtFace = false);
};
}  // namespace ablate::finiteVolume::boundaryConditions
#endif  // ABLATELIBRARY_ESSENTIALGHOST_HPP
