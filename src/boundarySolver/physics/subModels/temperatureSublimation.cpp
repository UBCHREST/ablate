#include "temperatureSublimation.hpp"
ablate::boundarySolver::physics::subModels::TemperatureSublimation::TemperatureSublimation(const std::shared_ptr<ablate::parameters::Parameters>& properties,
                                                                                           const std::shared_ptr<ablate::mathFunctions::MathFunction>& initialization,
                                                                                           const std::shared_ptr<ablate::parameters::Parameters>& options)
    : properties(properties),
      initialization(initialization),
      options(options),
      latentHeatOfFusion(properties->GetExpect<PetscReal>("latentHeatOfFusion")),
      solidDensity(properties->GetExpect<PetscReal>("density"))

{}

void ablate::boundarySolver::physics::subModels::TemperatureSublimation::Initialize(ablate::boundarySolver::BoundarySolver& bSolver) {
    SublimationModel::Initialize(bSolver);

    // Get the surface temperature from the properties
    auto sublimationTemperature = properties->GetExpect<double>("sublimationTemperature");

    /** Initialize the solid boundary heat transfer model */
    for (const auto& geom : bSolver.GetBoundaryGeometry()) {
        oneDimensionHeatTransfer[geom.geometry.faceId] = std::make_shared<OneDimensionHeatTransfer>(properties, initialization, options, sublimationTemperature);
        heatFluxIntoSolid[geom.geometry.faceId] = 0.0;
    }
}
PetscErrorCode ablate::boundarySolver::physics::subModels::TemperatureSublimation::Update(PetscInt faceId, PetscReal dt, PetscReal heatFluxToSurface, PetscReal& temperature) {
    PetscFunctionBegin;

    // Step the time stepper in time
    oneDimensionHeatTransfer[faceId]->Solve(heatFluxToSurface, dt, temperature, heatFluxIntoSolid[faceId]);

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ablate::boundarySolver::physics::subModels::TemperatureSublimation::Compute(PetscInt faceId, PetscReal heatFluxToSurface,
                                                                                           ablate::boundarySolver::physics::subModels::SublimationModel::SurfaceState& surfaceState) {
    PetscFunctionBeginHot;
    // compute the heat flux. Add the radiation heat flux for this face intensity if the radiation solver exists
    PetscReal sublimationHeatFlux = heatFluxToSurface - heatFluxIntoSolid[faceId];

    // We can only use positive heat flux
    sublimationHeatFlux = PetscMax(0.0, sublimationHeatFlux);

    // Compute the massFlux (we can only remove mass)
    surfaceState.massFlux = PetscMax(0.0, sublimationHeatFlux / latentHeatOfFusion);
    surfaceState.regressionRate = surfaceState.massFlux / solidDensity;

    PetscFunctionReturn(PETSC_SUCCESS);
}

#include "registrar.hpp"
REGISTER(ablate::boundarySolver::physics::subModels::SublimationModel, ablate::boundarySolver::physics::subModels::TemperatureSublimation,
         "Sublimation occurs at the specified temperature.  Extra heatFlux is used to heat the solid boundary",
         ARG(ablate::parameters::Parameters, "properties", "the heat transfer properties (specificHeat, conductivity, density, sublimationTemperature, latentHeatOfFusion"),
         ARG(ablate::mathFunctions::MathFunction, "initialization", " math function to initialize the temperature"),
         OPT(ablate::parameters::Parameters, "options", "the petsc options for the solver/ts")

);
