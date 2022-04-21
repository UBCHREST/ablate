#include "sublimation.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/processes/flowProcess.hpp"
#include "utilities/mathUtilities.hpp"

using fp = ablate::finiteVolume::CompressibleFlowFields;

ablate::boundarySolver::physics::Sublimation::Sublimation(PetscReal latentHeatOfFusion, PetscReal effectiveConductivity, std::shared_ptr<ablate::mathFunctions::FieldFunction> massFractions,
                                                          std::shared_ptr<mathFunctions::MathFunction> additionalHeatFlux)
    : latentHeatOfFusion(latentHeatOfFusion),
      effectiveConductivity(effectiveConductivity),
      additionalHeatFlux(additionalHeatFlux),
      massFractions(massFractions),
      massFractionsFunction(massFractions ? massFractions->GetFieldFunction()->GetPetscFunction() : nullptr),
      massFractionsContext(massFractions ? massFractions->GetFieldFunction()->GetContext() : nullptr) {}

void ablate::boundarySolver::physics::Sublimation::Initialize(ablate::boundarySolver::BoundarySolver &bSolver) {
    // check for species
    if (bSolver.GetSubDomain().ContainsField(finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD)) {
        bSolver.RegisterFunction(SublimationFunction,
                                 this,
                                 {finiteVolume::CompressibleFlowFields::EULER_FIELD, finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD},
                                 {finiteVolume::CompressibleFlowFields::EULER_FIELD, finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD},
                                 {finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD},
                                 BoundarySolver::BoundarySourceType::Distributed);

        numberSpecies = bSolver.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD).numberComponents;

    } else {
        bSolver.RegisterFunction(SublimationFunction,
                                 this,
                                 {finiteVolume::CompressibleFlowFields::EULER_FIELD},
                                 {finiteVolume::CompressibleFlowFields::EULER_FIELD},
                                 {finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD},
                                 BoundarySolver::BoundarySourceType::Distributed);
    }

    // If there is a additionalHeatFlux function, we need to update time
    if (additionalHeatFlux || massFractions) {
        bSolver.RegisterPreStep([this](auto ts, auto &solver) {
            PetscFunctionBeginUser;
            PetscErrorCode ierr = TSGetTime(ts, &(this->currentTime));
            CHKERRQ(ierr);

            PetscFunctionReturn(0);
        });
    }
}

PetscErrorCode ablate::boundarySolver::physics::Sublimation::SublimationFunction(PetscInt dim, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom *fg,
                                                                                 const PetscFVCellGeom *boundaryCell, const PetscInt *uOff, const PetscScalar *boundaryValues,
                                                                                 const PetscScalar **stencilValues, const PetscInt *aOff, const PetscScalar *auxValues,
                                                                                 const PetscScalar **stencilAuxValues, PetscInt stencilSize, const PetscInt *stencil, const PetscScalar *stencilWeights,
                                                                                 const PetscInt *sOff, PetscScalar *source, void *ctx) {
    PetscFunctionBeginUser;
    // Mark the locations
    const int EULER_LOC = 0;
    const int DENSITY_YI_LOC = 1;
    const int TEMPERATURE_LOC = 0;
    auto sublimation = (Sublimation *)ctx;

    // extract the temperature
    std::vector<PetscReal> stencilTemperature(stencilSize, 0);
    for (PetscInt s = 0; s < stencilSize; s++) {
        stencilTemperature[s] = stencilAuxValues[s][aOff[TEMPERATURE_LOC]];
    }

    // compute dTdn
    PetscReal dTdn;
    BoundarySolver::ComputeGradientAlongNormal(dim, fg, auxValues[aOff[TEMPERATURE_LOC]], stencilSize, stencilTemperature.data(), stencilWeights, dTdn);

    // compute the heat flux
    PetscReal heatFluxIntoSolid = PetscMax(0.0, -dTdn * sublimation->effectiveConductivity);  // note that q = -dTdn as dTdN faces into the solid

    // If there is an additional heat flux compute and add value
    if (sublimation->additionalHeatFlux) {
        heatFluxIntoSolid += sublimation->additionalHeatFlux->Eval(fg->centroid, (int)dim, sublimation->currentTime);
    }

    // Compute the massFlux (we can only remove mass)
    PetscReal massFlux = heatFluxIntoSolid / sublimation->latentHeatOfFusion;

    // Compute the area
    PetscReal area = utilities::MathUtilities::MagVector(dim, fg->areas);

    // Add the source term, kg/s for rho
    source[sOff[EULER_LOC] + fp::RHO] = massFlux * area;

    // Add each momentum flux
    PetscReal momentumFlux = massFlux * massFlux / boundaryValues[uOff[EULER_LOC] + fp::RHO];
    // And the mom flux for each dir by g
    for (PetscInt dir = 0; dir < dim; dir++) {
        source[sOff[EULER_LOC] + fp::RHOU + dir] = momentumFlux * -fg->areas[dir];
    }

    // Energy term
    source[sOff[EULER_LOC] + fp::RHOE] = -massFlux * sublimation->latentHeatOfFusion * area;

    // Add in species
    if (sublimation->massFractionsContext) {
        // Fill the source with the mass fractions
        PetscErrorCode ierr =
            sublimation->massFractionsFunction(dim, sublimation->currentTime, fg->centroid, sublimation->numberSpecies, source + sOff[DENSITY_YI_LOC], sublimation->massFractionsContext);
        CHKERRQ(ierr);

        // Scale the mass fractions by massFlux*area
        for (PetscInt sp = 0; sp < sublimation->numberSpecies; sp++) {
            source[sOff[DENSITY_YI_LOC] + sp] *= massFlux * area;
        }
    }

    PetscFunctionReturn(0);
}
void ablate::boundarySolver::physics::Sublimation::Initialize(PetscInt numberSpeciesIn) { numberSpecies = numberSpeciesIn; }

#include "registrar.hpp"
REGISTER(ablate::boundarySolver::BoundaryProcess, ablate::boundarySolver::physics::Sublimation, "Adds in the euler/yi sources for a sublimating material.  Should be used with a LODI boundary.",
         ARG(double, "latentHeatOfFusion", "the latent heat of fusion [J/kg]"), ARG(double, "effectiveConductivity", "the effective conductivity to compute heat flux to the surface [W/(m⋅K)]"),
         OPT(ablate::mathFunctions::FieldFunction, "massFractions", "the species to deposit the off gas mass to (required if solving species)"),
         OPT(ablate::mathFunctions::MathFunction, "additionalHeatFlux", "additional normal heat flux into the solid function"));
