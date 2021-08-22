#ifndef ABLATELIBRARY_ZEROGRADIENTGHOST_HPP
#define ABLATELIBRARY_ZEROGRADIENTGHOST_HPP

#include <mathFunctions/fieldFunction.hpp>
#include "ghost.hpp"

namespace ablate::flow::boundaryConditions {

class ZeroGradientGhost : public Ghost {
   private:
    static PetscErrorCode ZeroGradientGhostUpdate(PetscReal time, const PetscReal* c, const PetscReal* n, const PetscScalar* a_xI, PetscScalar* a_xG, void* ctx);

   public:
    ZeroGradientGhost(std::string fieldName, std::string boundaryName, std::vector<int> labelId, std::string labelName = {});
};
}  // namespace ablate::flow::boundaryConditions
#endif  // ABLATELIBRARY_ESSENTIALGHOST_HPP
