#include "radiationLoss.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"

ablate::finiteVolume::processes::RadiationLoss::RadiationLoss(std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn) : radiationModel(std::move(radiationModelIn)) {}

void ablate::finiteVolume::processes::RadiationLoss::Setup(ablate::finiteVolume::FiniteVolumeSolver &fvmSolver) {
        // add the source function
        fvmSolver.RegisterRHSFunction(ComputeRadiationLoss, &absorptivityFunction, {ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD}, {}, {ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD});

    absorptivityFunction = GetRadiationModel()->GetRadiationPropertiesTemperatureFunction(eos::radiationProperties::RadiationProperty::Absorptivity, fvmSolver.GetSubDomain().GetFields());
}

PetscErrorCode ablate::finiteVolume::processes::RadiationLoss::ComputeRadiationLoss(PetscInt dim, PetscReal time, const PetscFVCellGeom *cg, const PetscInt *uOff, const PetscScalar *u,
                                                                                    const PetscInt *aOff, const PetscScalar *a, PetscScalar *f, void *ctx) {
    PetscFunctionBegin;
    auto absorptivityFunction = (eos::ThermodynamicTemperatureFunction*)ctx;
    auto absorptivityFunctionContext = absorptivityFunction->context.get();  //!< Get access to the absorption function
    double kappa = 1;                  //!< Absorptivity coefficient, property of each cell

    /**
     * Get the rhs values so that the absorption can be read out of the solution vector
     * Get the absorption out of the solution vector
     */
    absorptivityFunction->function(u, a[0], &kappa, absorptivityFunctionContext); //! Temperature is the first offset in the aux array

    /**
     * Add the computed intensity to the energy equation
     */
    f[ablate::finiteVolume::CompressibleFlowFields::RHOE] += GetIntensity(a[0], kappa);  //!< Loop through the cells and update the equation of state
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
#define COMMA ,
REGISTER_PASS_THROUGH(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::RadiationLoss, "uses math functions to add arbitrary sources to the fvm method",
                      ablate::eos::radiationProperties::RadiationModel);
