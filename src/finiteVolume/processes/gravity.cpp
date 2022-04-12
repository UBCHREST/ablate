#include "gravity.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
ablate::finiteVolume::processes::Gravity::Gravity(std::vector<double> gravityVector) : gravityVector(gravityVector) {}

void ablate::finiteVolume::processes::Gravity::Initialize(ablate::finiteVolume::FiniteVolumeSolver &fv) {
    // add the source function
    fv.RegisterRHSFunction(ComputeGravitySource, this, {CompressibleFlowFields::EULER_FIELD}, {CompressibleFlowFields::EULER_FIELD}, {});
}

PetscErrorCode ablate::finiteVolume::processes::Gravity::ComputeGravitySource(PetscInt dim, PetscReal time, const PetscFVCellGeom *cg, const PetscInt *uOff, const PetscScalar *u,
                                                                              const PetscScalar *const *gradU, const PetscInt *aOff, const PetscScalar *a, const PetscScalar *const *gradA,
                                                                              PetscScalar *f, void *ctx) {
    PetscFunctionBeginUser;
    const int EULER_FIELD = 0;
    auto gravityProcess = (ablate::finiteVolume::processes::Gravity *)ctx;

    // exact some values
    const PetscReal density = u[uOff[EULER_FIELD] + CompressibleFlowFields::RHO];

    // set the source terms
    f[CompressibleFlowFields::RHO] = 0.0;
    f[CompressibleFlowFields::RHOE] = 0.0;

    // Add in the gravity source terms for momentum and energy
    for (PetscInt n = 0; n < dim; n++) {
        f[CompressibleFlowFields::RHOU + n] = density * gravityProcess->gravityVector[n];
        PetscReal vel = u[uOff[EULER_FIELD] + CompressibleFlowFields::RHOU + n] / density;
        f[CompressibleFlowFields::RHOE] += vel * f[CompressibleFlowFields::RHOU + n];
    }

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::Gravity, "build advection/diffusion for the euler field",
         ARG(std::vector<double>, "vector", "gravitational acceleration vector"));
