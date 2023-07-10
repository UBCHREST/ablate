#include "completeSublimation.hpp"
ablate::boundarySolver::subModels::CompleteSublimation::CompleteSublimation(PetscReal latentHeatOfFusion, PetscReal solidDensity)
    : latentHeatOfFusion(latentHeatOfFusion), solidDensity(solidDensity == 0.0 ? 1.0 : solidDensity) {}

PetscErrorCode ablate::boundarySolver::subModels::CompleteSublimation::Solve(PetscInt faceId, PetscReal heatFluxToSurface,
                                                                             ablate::boundarySolver::subModels::SublimationModel::SurfaceState& surfaceState) {
    PetscFunctionBeginHot;
    surfaceState.massFlux = PetscMax(0.0, heatFluxToSurface / latentHeatOfFusion);
    surfaceState.regressionRate = surfaceState.massFlux / solidDensity;

    PetscFunctionReturn(PETSC_SUCCESS);
}

#include "registrar.hpp"
REGISTER_DEFAULT(ablate::boundarySolver::subModels::SublimationModel, ablate::boundarySolver::subModels::CompleteSublimation, "Assumes all energy into the surface results in sublimation",
                 ARG(double, "latentHeatOfFusion", "the latent heat of fusion [J/kg]"),
                 OPT(double, "solidDensity", "Solid density of the fuel.  This is only used to output/report the solid regression rate. (Default is 1.0)"));
