#ifndef ABLATELIBRARY_DRAGMODEL_HPP
#define ABLATELIBRARY_DRAGMODEL_HPP

#include <petsc.h>

namespace ablate::particles::processes::drag {

class DragModel {
   public:
    virtual void ComputeDragForce(PetscInt dim, const PetscReal *partVel, const PetscReal *flowVel, PetscReal muF, PetscReal rhoF, PetscReal partDiam, PetscReal *dragForce) = 0;

    virtual ~DragModel() = default;
};

}  // namespace ablate::particles::processes::drag

#endif  // ABLATELIBRARY_DRAGMODEL_HPP
