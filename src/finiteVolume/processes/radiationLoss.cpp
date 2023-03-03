#include "radiationLoss.hpp"

ablate::finiteVolume::processes::RadiationLoss::RadiationLoss(std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn, double tInfinityIn)
    : radiationModel(std::move(radiationModelIn)), tInfinity(tInfinityIn == 0 ? 300 : tInfinityIn) {}

void ablate::finiteVolume::processes::RadiationLoss::Setup(ablate::finiteVolume::FiniteVolumeSolver &fvmSolver) {
    // add the source function
    fvmSolver.RegisterRHSFunction(ComputeRadiationLoss, this, {ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD}, {}, {ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD});

    absorptivityFunction = GetRadiationModel()->GetAbsorptionPropertiesTemperatureFunction(eos::radiationProperties::RadiationProperty::Absorptivity, fvmSolver.GetSubDomain().GetFields());
}

PetscErrorCode ablate::finiteVolume::processes::RadiationLoss::ComputeRadiationLoss(PetscInt dim, PetscReal time, const PetscFVCellGeom *cg, const PetscInt *uOff, const PetscScalar *u,
                                                                                    const PetscInt *aOff, const PetscScalar *a, PetscScalar *f, void *ctx) {
    PetscFunctionBegin;
    auto radiationLoss = (RadiationLoss *)ctx;
    auto absorptivityFunction = radiationLoss->absorptivityFunction;
    auto absorptivityFunctionContext = absorptivityFunction.context.get();  //!< Get access to the absorption function
    double kappa = 1;                                                       //!< Absorptivity coefficient, property of each cell

    /**
     * Get the rhs values so that the absorption can be read out of the solution vector
     * Get the absorption out of the solution vector
     */
    absorptivityFunction.function(u, a[aOff[0]], &kappa, absorptivityFunctionContext);  //! Temperature is the first offset in the aux array

    /**
     * Add the computed intensity to the energy equation
     */
    if ((GetIntensity(radiationLoss->tInfinity, a[aOff[0]], kappa) != 0) && (a[aOff[0]] < 300)) {
        PetscPrintf(PETSC_COMM_WORLD, "%f %f %f\n", GetIntensity(radiationLoss->tInfinity, a[aOff[0]], kappa), a[aOff[0]], kappa);
    }
    for (int d = 0; d < dim; d++) f[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = 0;
    f[ablate::finiteVolume::CompressibleFlowFields::RHO] = 0;
    f[ablate::finiteVolume::CompressibleFlowFields::RHOE] = GetIntensity(radiationLoss->tInfinity, a[aOff[0]], kappa);  //!< Loop through the cells and update the equation of state
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER_DEFAULT(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::RadiationLoss,
                 "Computes radiative losses for a volumetric region without evaluating the gains via raytracing",
                 ARG(ablate::eos::radiationProperties::RadiationModel, "properties", "the radiation properties model"),
                 OPT(double, "tInfinity", "far field temperature of the domain when computing radiation losses without ray tracing gains (default 300 K)"));
