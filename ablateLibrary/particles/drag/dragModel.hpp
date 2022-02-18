#ifndef ABLATELIBRARY_DRAGMODEL_HPP
#define ABLATELIBRARY_DRAGMODEL_HPP

#include <petsc.h>

namespace ablate::particles::drag {

class DragModel {
   public:
    virtual void ComputeDragForce(const PetscInt dim, const PetscReal *partVel, const PetscReal *flowVel, const PetscReal muF, const PetscReal rhoF, const PetscReal partDiam,
                                  PetscReal *dragForce) = 0;

    virtual ~DragModel() = default;
};

}  // namespace ablate::particles::drag

#endif  // ABLATELIBRARY_DRAGMODEL_HPP
