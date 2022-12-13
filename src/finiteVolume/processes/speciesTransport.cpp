#include "speciesTransport.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/mathUtilities.hpp"

ablate::finiteVolume::processes::SpeciesTransport::SpeciesTransport(std::shared_ptr<eos::EOS> eosIn, std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalcIn,
                                                                    std::shared_ptr<eos::transport::TransportModel> transportModelIn)
    : fluxCalculator(std::move(fluxCalcIn)), eos(std::move(eosIn)), transportModel(std::move(transportModelIn)), advectionData() {
    if (fluxCalculator) {
        // set the decode state function
        advectionData.numberSpecies = (PetscInt)eos->GetSpecies().size();

        // extract the difference function from fluxDifferencer object
        advectionData.fluxCalculatorFunction = fluxCalculator->GetFluxCalculatorFunction();
        advectionData.fluxCalculatorCtx = fluxCalculator->GetFluxCalculatorContext();
    }

    if (transportModel) {
        // set the eos functions
        diffusionData.numberSpecies = (PetscInt)eos->GetSpecies().size();
        diffusionData.speciesSpeciesSensibleEnthalpy.resize(eos->GetSpecies().size());
    }

    numberSpecies = (PetscInt)eos->GetSpecies().size();
}

void ablate::finiteVolume::processes::SpeciesTransport::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    if (!eos->GetSpeciesVariables().empty()) {
        if (fluxCalculator) {
            flow.RegisterRHSFunction(AdvectionFlux, &advectionData, CompressibleFlowFields::DENSITY_YI_FIELD, {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD}, {});
            advectionData.computeTemperature = eos->GetThermodynamicFunction(eos::ThermodynamicProperty::Temperature, flow.GetSubDomain().GetFields());
            advectionData.computeInternalEnergy = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::InternalSensibleEnergy, flow.GetSubDomain().GetFields());
            advectionData.computeSpeedOfSound = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SpeedOfSound, flow.GetSubDomain().GetFields());
            advectionData.computePressure = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Pressure, flow.GetSubDomain().GetFields());
        }

        if (transportModel) {
            diffusionData.diffFunction = transportModel->GetTransportTemperatureFunction(eos::transport::TransportProperty::Diffusivity, flow.GetSubDomain().GetFields());
            if (diffusionData.diffFunction.function) {
                flow.RegisterRHSFunction(DiffusionEnergyFlux,
                                         &diffusionData,
                                         CompressibleFlowFields::EULER_FIELD,
                                         {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD},
                                         {CompressibleFlowFields::YI_FIELD});
                flow.RegisterRHSFunction(DiffusionSpeciesFlux,
                                         &diffusionData,
                                         CompressibleFlowFields::DENSITY_YI_FIELD,
                                         {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD},
                                         {CompressibleFlowFields::YI_FIELD});

                diffusionData.computeTemperatureFunction = eos->GetThermodynamicFunction(eos::ThermodynamicProperty::Temperature, flow.GetSubDomain().GetFields());
                diffusionData.computeSpeciesSensibleEnthalpyFunction = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SpeciesSensibleEnthalpy, flow.GetSubDomain().GetFields());
            }
        }

        flow.RegisterAuxFieldUpdate(
            UpdateAuxMassFractionField, &numberSpecies, std::vector<std::string>{CompressibleFlowFields::YI_FIELD}, {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD});

        // clean up the species
        flow.RegisterPostEvaluate(NormalizeSpecies);
    }
}

PetscErrorCode ablate::finiteVolume::processes::SpeciesTransport::UpdateAuxMassFractionField(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[],
                                                                                             const PetscScalar *conservedValues, const PetscInt aOff[], PetscScalar *auxField, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[uOff[0] + CompressibleFlowFields::RHO];

    auto numberSpecies = (PetscInt *)ctx;

    for (PetscInt sp = 0; sp < *numberSpecies; sp++) {
        auxField[aOff[0] + sp] = conservedValues[uOff[1] + sp] / density;
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::SpeciesTransport::DiffusionEnergyFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscInt uOff_x[],
                                                                                      const PetscScalar field[], const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[],
                                                                                      const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void *ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int yi = 0;
    const int euler = 0;

    auto flowParameters = (DiffusionData *)ctx;

    // get the current density from euler
    const PetscReal density = field[uOff[euler] + CompressibleFlowFields::RHO];

    // compute the temperature in this volume
    PetscErrorCode ierr;
    PetscReal temperature;
    ierr = flowParameters->computeTemperatureFunction.function(field, &temperature, flowParameters->computeTemperatureFunction.context.get());
    CHKERRQ(ierr);

    ierr = flowParameters->computeSpeciesSensibleEnthalpyFunction.function(
        field, temperature, flowParameters->speciesSpeciesSensibleEnthalpy.data(), flowParameters->computeSpeciesSensibleEnthalpyFunction.context.get());
    CHKERRQ(ierr);

    // set the non rho E fluxes to zero
    flux[CompressibleFlowFields::RHO] = 0.0;
    flux[CompressibleFlowFields::RHOE] = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        flux[CompressibleFlowFields::RHOU + d] = 0.0;
    }

    // compute diff
    PetscReal diff = 0.0;
    flowParameters->diffFunction.function(field, temperature, &diff, flowParameters->diffFunction.context.get());

    for (PetscInt sp = 0; sp < flowParameters->numberSpecies; ++sp) {
        for (PetscInt d = 0; d < dim; ++d) {
            // speciesFlux(-rho Di dYi/dx - rho Di dYi/dy - rho Di dYi//dz) . n A
            const int offset = aOff_x[yi] + (sp * dim) + d;
            PetscReal speciesFlux = -fg->normal[d] * density * diff * flowParameters->speciesSpeciesSensibleEnthalpy[sp] * gradAux[offset];
            flux[CompressibleFlowFields::RHOE] += speciesFlux;
        }
    }

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::finiteVolume::processes::SpeciesTransport::DiffusionSpeciesFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscInt uOff_x[],
                                                                                       const PetscScalar field[], const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[],
                                                                                       const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void *ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int yi = 0;
    const int euler = 0;

    auto flowParameters = (DiffusionData *)ctx;

    // get the current density from euler
    const PetscReal density = field[uOff[euler] + CompressibleFlowFields::RHO];

    PetscErrorCode ierr;
    PetscReal temperature;
    ierr = flowParameters->computeTemperatureFunction.function(field, &temperature, flowParameters->computeTemperatureFunction.context.get());
    CHKERRQ(ierr);

    // compute diff
    PetscReal diff = 0.0;
    flowParameters->diffFunction.function(field, temperature, &diff, flowParameters->diffFunction.context.get());

    // species equations
    for (PetscInt sp = 0; sp < flowParameters->numberSpecies; ++sp) {
        flux[sp] = 0;
        for (PetscInt d = 0; d < dim; ++d) {
            // speciesFlux(-rho Di dYi/dx - rho Di dYi/dy - rho Di dYi//dz) . n A
            const int offset = aOff_x[yi] + (sp * dim) + d;
            PetscReal speciesFlux = -fg->normal[d] * density * diff * gradAux[offset];
            flux[sp] += speciesFlux;
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::SpeciesTransport::AdvectionFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscScalar *fieldL, const PetscScalar *fieldR,
                                                                                const PetscInt *aOff, const PetscScalar *auxL, const PetscScalar *auxR, PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
    auto eulerAdvectionData = (AdvectionData *)ctx;

    // Compute the norm
    PetscReal norm[3];
    utilities::MathUtilities::NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = utilities::MathUtilities::MagVector(dim, fg->normal);

    const int EULER_FIELD = 0;
    const int YI_FIELD = 1;

    // Decode the left and right states
    PetscReal densityL;
    PetscReal normalVelocityL;
    PetscReal velocityL[3];
    PetscReal internalEnergyL;
    PetscReal aL;
    PetscReal pL;
    // decode the left side
    {
        densityL = fieldL[uOff[EULER_FIELD] + CompressibleFlowFields::RHO];
        PetscReal temperatureL;

        PetscErrorCode ierr = eulerAdvectionData->computeTemperature.function(fieldL, &temperatureL, eulerAdvectionData->computeTemperature.context.get());
        CHKERRQ(ierr);

        // Get the velocity in this direction
        normalVelocityL = 0.0;
        for (PetscInt d = 0; d < dim; d++) {
            velocityL[d] = fieldL[uOff[EULER_FIELD] + CompressibleFlowFields::RHOU + d] / densityL;
            normalVelocityL += velocityL[d] * norm[d];
        }

        ierr = eulerAdvectionData->computeInternalEnergy.function(fieldL, temperatureL, &internalEnergyL, eulerAdvectionData->computeInternalEnergy.context.get());
        CHKERRQ(ierr);
        ierr = eulerAdvectionData->computeSpeedOfSound.function(fieldL, temperatureL, &aL, eulerAdvectionData->computeSpeedOfSound.context.get());
        CHKERRQ(ierr);
        ierr = eulerAdvectionData->computePressure.function(fieldL, temperatureL, &pL, eulerAdvectionData->computePressure.context.get());
        CHKERRQ(ierr);
    }

    PetscReal densityR;
    PetscReal normalVelocityR;
    PetscReal velocityR[3];
    PetscReal internalEnergyR;
    PetscReal aR;
    PetscReal pR;
    {  // decode right state
        densityR = fieldR[uOff[EULER_FIELD] + CompressibleFlowFields::RHO];
        PetscReal temperatureR;

        PetscErrorCode ierr = eulerAdvectionData->computeTemperature.function(fieldR, &temperatureR, eulerAdvectionData->computeTemperature.context.get());
        CHKERRQ(ierr);

        // Get the velocity in this direction
        normalVelocityR = 0.0;
        for (PetscInt d = 0; d < dim; d++) {
            velocityR[d] = fieldR[uOff[EULER_FIELD] + CompressibleFlowFields::RHOU + d] / densityR;
            normalVelocityR += velocityR[d] * norm[d];
        }

        ierr = eulerAdvectionData->computeInternalEnergy.function(fieldR, temperatureR, &internalEnergyR, eulerAdvectionData->computeInternalEnergy.context.get());
        CHKERRQ(ierr);
        ierr = eulerAdvectionData->computeSpeedOfSound.function(fieldR, temperatureR, &aR, eulerAdvectionData->computeSpeedOfSound.context.get());
        CHKERRQ(ierr);
        ierr = eulerAdvectionData->computePressure.function(fieldR, temperatureR, &pR, eulerAdvectionData->computePressure.context.get());
        CHKERRQ(ierr);
    }

    // get the face values
    PetscReal massFlux;

    if (eulerAdvectionData->fluxCalculatorFunction(eulerAdvectionData->fluxCalculatorCtx, normalVelocityL, aL, densityL, pL, normalVelocityR, aR, densityR, pR, &massFlux, nullptr) ==
        fluxCalculator::LEFT) {
        // march over each gas species
        for (PetscInt sp = 0; sp < eulerAdvectionData->numberSpecies; sp++) {
            // Note: there is no density in the flux because uR and UL are density*yi
            flux[sp] = (massFlux * fieldL[uOff[YI_FIELD] + sp] / densityL) * areaMag;
        }
    } else {
        // march over each gas species
        for (PetscInt sp = 0; sp < eulerAdvectionData->numberSpecies; sp++) {
            // Note: there is no density in the flux because uR and UL are density*yi
            flux[sp] = (massFlux * fieldR[uOff[YI_FIELD] + sp] / densityR) * areaMag;
        }
    }

    PetscFunctionReturn(0);
}

void ablate::finiteVolume::processes::SpeciesTransport::NormalizeSpecies(TS ts, ablate::solver::Solver &solver) {
    // Get the density and densityYi field info
    const auto &eulerFieldInfo = solver.GetSubDomain().GetField(CompressibleFlowFields::EULER_FIELD);
    const auto &densityYiFieldInfo = solver.GetSubDomain().GetField(CompressibleFlowFields::DENSITY_YI_FIELD);

    // Get the solution vec and dm
    auto dm = solver.GetSubDomain().GetDM();
    auto solVec = solver.GetSubDomain().GetSolutionVector();

    // Get the array vector
    PetscScalar *solutionArray;
    VecGetArray(solVec, &solutionArray) >> checkError;

    // March over each cell in this domain
    solver::Range cellRange;
    solver.GetCellRange(cellRange);

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.points ? cellRange.points[c] : c;

        // Get the euler and density field
        const PetscScalar *euler = nullptr;
        DMPlexPointGlobalFieldRead(dm, cell, eulerFieldInfo.id, solutionArray, &euler) >> checkError;
        PetscScalar *densityYi;
        DMPlexPointGlobalFieldRef(dm, cell, densityYiFieldInfo.id, solutionArray, &densityYi) >> checkError;

        // Only update if in the global vector
        if (euler) {
            // Get density
            const PetscScalar density = euler[CompressibleFlowFields::RHO];

            PetscScalar yiSum = 0.0;
            for (PetscInt sp = 0; sp < densityYiFieldInfo.numberComponents - 1; sp++) {
                // Limit the bounds
                PetscScalar yi = densityYi[sp] / density;
                yi = PetscMax(0.0, yi);
                yi = PetscMin(1.0, yi);
                yiSum += yi;

                // Set it back
                densityYi[sp] = yi * density;
            }

            // Now cleanup yi
            if (yiSum > 1.0) {
                for (PetscInt sp = 0; sp < densityYiFieldInfo.numberComponents; sp++) {
                    PetscScalar yi = densityYi[sp] / density;
                    yi /= yiSum;
                    densityYi[sp] = density * yi;
                }
                densityYi[densityYiFieldInfo.numberComponents - 1] = 0.0;
            } else {
                densityYi[densityYiFieldInfo.numberComponents - 1] = density * (1.0 - yiSum);
            }
        }
    }

    // cleanup
    VecRestoreArray(solVec, &solutionArray) >> checkError;
    solver.RestoreRange(cellRange);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::SpeciesTransport, "diffusion/advection for the species yi field",
         ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"),
         OPT(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculator", "the flux calculator (default is no advection)"),
         OPT(ablate::eos::transport::TransportModel, "transport", "the diffusion transport model (default is no diffusion)"));
