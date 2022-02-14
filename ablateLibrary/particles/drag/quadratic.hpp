#ifndef ABLATELIBRARY_QUADRATIC_HPP
#define ABLATELIBRARY_QUADRATIC_HPP

#include "dragModel.hpp"

namespace ablate::particles::drag {

class Quadratic : public DragModel {
   public:
    void ComputeDragForce(const PetscInt dim, const PetscReal *partVel, const PetscReal *flowVel, const PetscReal muF, const PetscReal rhoF, const PetscReal partDiam, const PetscReal *dragForce) override;
};

}  // namespace ablate::particles::drag

#endif  // ABLATELIBRARY_QUADRATIC_HPP
