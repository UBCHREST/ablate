#include "sublimation.hpp"
#include <utility>
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/mathUtilities.hpp"

using fp = ablate::finiteVolume::CompressibleFlowFields;

ablate::boundarySolver::physics::Sublimation::Sublimation(PetscReal latentHeatOfFusion, std::shared_ptr<ablate::eos::transport::TransportModel> transportModel, std::shared_ptr<ablate::eos::EOS> eos,
                                                          const std::shared_ptr<ablate::mathFunctions::FieldFunction> &massFractions, std::shared_ptr<mathFunctions::MathFunction> additionalHeatFlux,
                                                          std::shared_ptr<finiteVolume::processes::PressureGradientScaling> pressureGradientScaling, bool disablePressure)
    : latentHeatOfFusion(latentHeatOfFusion),
      transportModel(std::move(transportModel)),
      eos(std::move(eos)),
      additionalHeatFlux(std::move(additionalHeatFlux)),
      massFractions(massFractions),
      massFractionsFunction(massFractions ? massFractions->GetFieldFunction()->GetPetscFunction() : nullptr),
      massFractionsContext(massFractions ? massFractions->GetFieldFunction()->GetContext() : nullptr),
      disablePressure(disablePressure),
      pressureGradientScaling(std::move(pressureGradientScaling)) {}

void ablate::boundarySolver::physics::Sublimation::Initialize(ablate::boundarySolver::BoundarySolver &bSolver) {
    // check for species
    std::vector<std::string> inputFields = {finiteVolume::CompressibleFlowFields::EULER_FIELD};
    // check for density yi field
    if (bSolver.GetSubDomain().ContainsField(finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD)) {
        inputFields.push_back(finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD);
    }

    // register the required sublimantion function
    bSolver.RegisterFunction(SublimationFunction, this, inputFields, inputFields, {finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD}, BoundarySolver::BoundarySourceType::Flux);

    // Register an optional output function
    bSolver.RegisterFunction(SublimationOutputFunction,
                             this,
                             {"conduction", "extraRad", "regressionMassFlux"},
                             inputFields,
                             {finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD},
                             BoundarySolver::BoundarySourceType::Face);

    numberSpecies = bSolver.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD).numberComponents;

    // extract the effectiveConductivity and viscosity model
    if (transportModel) {
        effectiveConductivity = transportModel->GetTransportTemperatureFunction(eos::transport::TransportProperty::Conductivity, bSolver.GetSubDomain().GetFields());
        viscosityFunction = transportModel->GetTransportTemperatureFunction(eos::transport::TransportProperty::Viscosity, bSolver.GetSubDomain().GetFields());
    }

    // If there is a additionalHeatFlux function, we need to update time
    if (additionalHeatFlux || massFractions) {
        bSolver.RegisterPreStep([this](auto ts, auto &solver) {
            PetscFunctionBeginUser;
            PetscCall(TSGetTime(ts, &(this->currentTime)));
            PetscFunctionReturn(0);
        });
    }

    if (bSolver.GetSubDomain().ContainsField(finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD)) {
        // set decode state functions
        computeTemperatureFunction = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Temperature, bSolver.GetSubDomain().GetFields());
        // add in aux update variables
        bSolver.RegisterAuxFieldUpdate(ablate::finiteVolume::processes::NavierStokesTransport::UpdateAuxTemperatureField,
                                       &computeTemperatureFunction,
                                       std::vector<std::string>{finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD},
                                       {});
    }

    computeSensibleEnthalpy = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SensibleEnthalpy, bSolver.GetSubDomain().GetFields());
    computePressure = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Pressure, bSolver.GetSubDomain().GetFields());

    if (bSolver.GetSubDomain().ContainsField(finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD)) {
        bSolver.RegisterPreStep([this](auto ts, auto &solver) { UpdateSpecies(ts, solver); });
        if (!massFractionsFunction) {
            throw std::invalid_argument("The massFractions must be specified for ablate::boundarySolver::physics::Sublimation when DENSITY_YI_FIELD is active.");
        }
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

    // compute the effectiveConductivity
    PetscReal effectiveConductivity = 0.0;
    if (sublimation->effectiveConductivity.function) {
        PetscCall(sublimation->effectiveConductivity.function(boundaryValues, auxValues[aOff[TEMPERATURE_LOC]], &effectiveConductivity, sublimation->effectiveConductivity.context.get()));
    }

    // compute the heat flux
    PetscReal heatFluxIntoSolid = -dTdn * effectiveConductivity;
    PetscReal sublimationHeatFlux = PetscMax(0.0, heatFluxIntoSolid);  // note that q = -dTdn as dTdN faces into the solid
    // If there is an additional heat flux compute and add value
    if (sublimation->additionalHeatFlux) {
        sublimationHeatFlux += sublimation->additionalHeatFlux->Eval(fg->centroid, (int)dim, sublimation->currentTime);
    }

    // Compute the massFlux (we can only remove mass)
    PetscReal massFlux = sublimationHeatFlux / sublimation->latentHeatOfFusion;

    // Compute the area
    PetscReal area = utilities::MathUtilities::MagVector(dim, fg->areas);

    // Determine the velocity in the normal coordinate system [nx, nt, nt]
    PetscReal boundaryDensity = boundaryValues[uOff[EULER_LOC] + fp::RHO];
    PetscReal velocityNormSystem[3] = {-massFlux / boundaryDensity, 0.0, 0.0};  // note the minus sign because the normal points out of the domain

    // Map this velocity into cartesian system
    // Compute the transformation matrix
    PetscReal transformationMatrix[3][3];
    PetscReal velocityCartSystem[3];
    utilities::MathUtilities::ComputeTransformationMatrix(dim, fg->normal, transformationMatrix);
    ablate::utilities::MathUtilities::MultiplyTranspose(dim, transformationMatrix, velocityNormSystem, velocityCartSystem);

    // store the gradient of velocity so that it can be used with CompressibleFlowComputeStressTensor [dudx, dudy, dudz, dvdx ...]
    PetscReal gradBoundaryVelocity[9];
    PetscArrayzero(gradBoundaryVelocity, 9);

    // For each component of velocity
    for (PetscInt v = 0; v < dim; v++) {
        for (PetscInt s = 0; s < stencilSize; ++s) {
            PetscReal stencilDensity = stencilValues[s][uOff[EULER_LOC] + finiteVolume::CompressibleFlowFields::RHO];

            PetscScalar delta = stencilValues[s][uOff[EULER_LOC] + finiteVolume::CompressibleFlowFields::RHOU + v] / stencilDensity - velocityCartSystem[v];

            for (PetscInt d = 0; d < dim; ++d) {
                gradBoundaryVelocity[v * dim + d] += stencilWeights[s * dim + d] * delta;
            }
        }
    }

    // compute the effectiveConductivity
    PetscReal viscosity = 0.0;
    if (sublimation->viscosityFunction.function) {
        PetscCall(sublimation->viscosityFunction.function(boundaryValues, auxValues[aOff[TEMPERATURE_LOC]], &viscosity, sublimation->viscosityFunction.context.get()));
    }

    // Compute the stress tensor tau
    PetscReal tau[9];  // Maximum size without symmetry
    PetscCall(ablate::finiteVolume::processes::NavierStokesTransport::CompressibleFlowComputeStressTensor(dim, viscosity, gradBoundaryVelocity, tau));

    // Add the source term, kg/s for rho
    source[sOff[EULER_LOC] + fp::RHO] = massFlux * area;

    // Add each momentum flux
    PetscReal momentumFlux = massFlux * massFlux / boundaryDensity;

    // compute the pressure on the face.  The first pressure in the stencil is always the node pressure on the face
    PetscReal boundaryPressure = 0.0;
    if (!sublimation->disablePressure) {
        PetscCall(sublimation->computePressure.function(stencilValues[0], stencilAuxValues[0][aOff[TEMPERATURE_LOC]], &boundaryPressure, sublimation->computePressure.context.get()));
        if (sublimation->pressureGradientScaling) {
            boundaryPressure /= PetscSqr(sublimation->pressureGradientScaling->GetAlpha());
        }
    }

    // And the mom flux for each dir by g
    for (PetscInt dir = 0; dir < dim; dir++) {
        source[sOff[EULER_LOC] + fp::RHOU + dir] = momentumFlux * -fg->areas[dir] - boundaryPressure * fg->areas[dir];

        // March over each direction for the viscus flux
        for (PetscInt d = 0; d < dim; ++d) {
            source[sOff[EULER_LOC] + fp::RHOU + dir] += fg->areas[d] * tau[dir * dim + d];  // This is tau[c][d]
        }
    }

    // compute the sensible enthalpy
    PetscReal sensibleEnthalpy;
    PetscCall(sublimation->computeSensibleEnthalpy.function(boundaryValues, auxValues[aOff[TEMPERATURE_LOC]], &sensibleEnthalpy, sublimation->computeSensibleEnthalpy.context.get()));

    // Energy term
    source[sOff[EULER_LOC] + fp::RHOE] = (massFlux * sensibleEnthalpy - heatFluxIntoSolid) * area;

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

PetscErrorCode ablate::boundarySolver::physics::Sublimation::SublimationOutputFunction(PetscInt dim, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom *fg,
                                                                                       const PetscFVCellGeom *boundaryCell, const PetscInt *uOff, const PetscScalar *boundaryValues,
                                                                                       const PetscScalar **stencilValues, const PetscInt *aOff, const PetscScalar *auxValues,
                                                                                       const PetscScalar **stencilAuxValues, PetscInt stencilSize, const PetscInt *stencil,
                                                                                       const PetscScalar *stencilWeights, const PetscInt *sOff, PetscScalar *source, void *ctx) {
    PetscFunctionBeginUser;
    // Mark the locations
    const int TEMPERATURE_LOC = 0;
    auto sublimation = (Sublimation *)ctx;

    // Store indexes to the expected output variables
    const int CONDUCTION_LOC = 0;
    const int EXTRA_RAD_LOC = 1;
    const int REGRESSION_MASS_FLUX_LOC = 2;

    // extract the temperature
    std::vector<PetscReal> stencilTemperature(stencilSize, 0);
    for (PetscInt s = 0; s < stencilSize; s++) {
        stencilTemperature[s] = stencilAuxValues[s][aOff[TEMPERATURE_LOC]];
    }

    // compute dTdn
    PetscReal dTdn;
    BoundarySolver::ComputeGradientAlongNormal(dim, fg, auxValues[aOff[TEMPERATURE_LOC]], stencilSize, stencilTemperature.data(), stencilWeights, dTdn);

    // compute the effectiveConductivity
    PetscReal effectiveConductivity = 0.0;
    if (sublimation->effectiveConductivity.function) {
        PetscCall(sublimation->effectiveConductivity.function(boundaryValues, auxValues[aOff[TEMPERATURE_LOC]], &effectiveConductivity, sublimation->effectiveConductivity.context.get()));
    }

    // compute the heat flux
    PetscReal heatFluxIntoSolid = -dTdn * effectiveConductivity;
    source[sOff[CONDUCTION_LOC]] = heatFluxIntoSolid;
    PetscReal sublimationHeatFlux = PetscMax(0.0, heatFluxIntoSolid);  // note that q = -dTdn as dTdN faces into the solid
    // If there is an additional heat flux compute and add value
    PetscReal additionalHeatFlux = 0.0;
    if (sublimation->additionalHeatFlux) {
        additionalHeatFlux = sublimation->additionalHeatFlux->Eval(fg->centroid, (int)dim, sublimation->currentTime);
        sublimationHeatFlux += additionalHeatFlux;
    }
    source[sOff[EXTRA_RAD_LOC]] = additionalHeatFlux;

    // Compute the massFlux (we can only remove mass)
    source[sOff[REGRESSION_MASS_FLUX_LOC]] = sublimationHeatFlux / sublimation->latentHeatOfFusion;

    PetscFunctionReturn(0);
}

void ablate::boundarySolver::physics::Sublimation::Initialize(PetscInt numberSpeciesIn) {
    numberSpecies = numberSpeciesIn;
    // for test code, extract the effectiveConductivity model without any fields
    effectiveConductivity = transportModel->GetTransportTemperatureFunction(eos::transport::TransportProperty::Conductivity, {});
    viscosityFunction = transportModel->GetTransportTemperatureFunction(eos::transport::TransportProperty::Viscosity, {});
    computeSensibleEnthalpy = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SensibleEnthalpy, {});
    computePressure = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Pressure, {});
}

void ablate::boundarySolver::physics::Sublimation::UpdateSpecies(TS ts, ablate::solver::Solver &solver) {
    // Get the density and densityYi field info
    const auto &eulerFieldInfo = solver.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::EULER_FIELD);
    const auto &densityYiFieldInfo = solver.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD);
    const auto &yiFieldInfo = solver.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::YI_FIELD);

    // Get the solution vec and dm
    auto dm = solver.GetSubDomain().GetDM();
    auto solVec = solver.GetSubDomain().GetSolutionVector();
    auto auxDm = solver.GetSubDomain().GetAuxDM();
    auto auxVec = solver.GetSubDomain().GetAuxVector();

    // get the time from the ts
    PetscReal time;
    TSGetTime(ts, &time) >> checkError;

    // Get the array vector
    PetscScalar *solutionArray;
    VecGetArray(solVec, &solutionArray) >> checkError;
    PetscScalar *auxArray;
    VecGetArray(auxVec, &auxArray) >> checkError;

    // March over each cell in this domain
    solver::Range cellRange;
    solver.GetCellRange(cellRange);
    auto dim = solver.GetSubDomain().GetDimensions();

    // Get the cell geom vec
    Vec cellGeomVec;
    const PetscScalar *cellGeomArray;
    DM cellGeomDm;
    DMPlexGetDataFVM(dm, nullptr, &cellGeomVec, nullptr, nullptr) >> checkError;
    VecGetDM(cellGeomVec, &cellGeomDm) >> checkError;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.points ? cellRange.points[c] : c;

        // Get the euler and density field
        const PetscScalar *euler = nullptr;
        DMPlexPointGlobalFieldRead(dm, cell, eulerFieldInfo.id, solutionArray, &euler) >> checkError;
        PetscScalar *densityYi;
        DMPlexPointGlobalFieldRef(dm, cell, densityYiFieldInfo.id, solutionArray, &densityYi) >> checkError;
        PetscScalar *yi;
        DMPlexPointLocalFieldRead(auxDm, cell, yiFieldInfo.id, auxArray, &yi) >> checkError;
        PetscFVCellGeom *cellGeom;
        DMPlexPointLocalRead(cellGeomDm, cell, cellGeomArray, &cellGeom) >> checkError;

        // compute the mass fractions on the boundary
        massFractionsFunction(dim, time, cellGeom->centroid, yiFieldInfo.numberComponents, yi, massFractionsContext);

        // Only update if in the global vector
        if (euler) {
            // Get density
            const PetscScalar density = euler[finiteVolume::CompressibleFlowFields::RHO];

            for (PetscInt sp = 0; sp < densityYiFieldInfo.numberComponents; sp++) {
                densityYi[sp] = yi[sp] * density;
            }
        }
    }

    // cleanup
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    VecRestoreArray(auxVec, &auxArray) >> checkError;
    VecRestoreArray(solVec, &solutionArray) >> checkError;
    solver.RestoreRange(cellRange);
}

#include "registrar.hpp"
REGISTER(ablate::boundarySolver::BoundaryProcess, ablate::boundarySolver::physics::Sublimation, "Adds in the euler/yi sources for a sublimating material.  Should be used with a LODI boundary.",
         ARG(double, "latentHeatOfFusion", "the latent heat of fusion [J/kg]"),
         OPT(ablate::eos::transport::TransportModel, "transportModel", "the effective conductivity model to compute heat flux to the surface [W/(m⋅K)]"),
         ARG(ablate::eos::EOS, "eos", "the eos used to compute temperature on the boundary"),
         OPT(ablate::mathFunctions::FieldFunction, "massFractions", "the species to deposit the off gas mass to (required if solving species)"),
         OPT(ablate::mathFunctions::MathFunction, "additionalHeatFlux", "additional normal heat flux into the solid function"),
         OPT(ablate::finiteVolume::processes::PressureGradientScaling, "pgs", "Pressure gradient scaling is used to scale the acoustic propagation speed and increase time step for low speed flows"),
         OPT(bool, "disablePressure", "disables the pressure contribution to the momentum equation. Should be true when advection is not solved. (Default is false)"));
