#include "speciesTransport.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/mathUtilities.hpp"

ablate::finiteVolume::processes::SpeciesTransport::SpeciesTransport(std::shared_ptr<eos::EOS> eosIn, std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalcIn,
                                                                    std::shared_ptr<eos::transport::TransportModel> transportModelIn)
    : fluxCalculator(std::move(fluxCalcIn)), eos(std::move(eosIn)), transportModel(std::move(transportModelIn)), advectionData() {
    if (fluxCalculator) {
        // set the decode state function
        advectionData.decodeStateFunction = eos->GetDecodeStateFunction();
        advectionData.decodeStateContext = eos->GetDecodeStateContext();
        advectionData.numberSpecies = (PetscInt)eos->GetSpecies().size();

        // extract the difference function from fluxDifferencer object
        advectionData.fluxCalculatorFunction = fluxCalculator->GetFluxCalculatorFunction();
        advectionData.fluxCalculatorCtx = fluxCalculator->GetFluxCalculatorContext();
    }

    if (transportModel) {
        diffusionData.diffFunction = transportModel->GetComputeDiffusivityFunction();
        diffusionData.diffContext = transportModel->GetComputeDiffusivityContext();

        // set the eos functions
        diffusionData.numberSpecies = (PetscInt)eos->GetSpecies().size();
        diffusionData.computeTemperatureFunction = eos->GetComputeTemperatureFunction();
        diffusionData.computeTemperatureContext = eos->GetComputeTemperatureContext();

        diffusionData.computeSpeciesSensibleEnthalpyFunction = eos->GetComputeSpeciesSensibleEnthalpyFunction();
        diffusionData.computeSpeciesSensibleEnthalpyContext = eos->GetComputeSpeciesSensibleEnthalpyContext();
        diffusionData.speciesSpeciesSensibleEnthalpy.resize(eos->GetSpecies().size());
    } else {
        diffusionData.diffFunction = nullptr;
        diffusionData.diffContext = nullptr;
    }

    numberSpecies = (PetscInt)eos->GetSpecies().size();
}

void ablate::finiteVolume::processes::SpeciesTransport::Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    if (!eos->GetSpecies().empty()) {
        if (fluxCalculator) {
            flow.RegisterRHSFunction(AdvectionFlux, &advectionData, CompressibleFlowFields::DENSITY_YI_FIELD, {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD}, {});
        }

        if (diffusionData.diffFunction) {
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
        }

        flow.RegisterAuxFieldUpdate(UpdateAuxMassFractionField, &numberSpecies, CompressibleFlowFields::YI_FIELD, {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD});

        // clean up the species
        flow.RegisterPostEvaluate(NormalizeSpecies);
    }
}

PetscErrorCode ablate::finiteVolume::processes::SpeciesTransport::UpdateAuxMassFractionField(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[],
                                                                                             const PetscScalar *conservedValues, PetscScalar *auxField, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[uOff[0] + RHO];

    auto numberSpecies = (PetscInt *)ctx;

    for (PetscInt sp = 0; sp < *numberSpecies; sp++) {
        auxField[sp] = conservedValues[uOff[1] + sp] / density;
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::SpeciesTransport::DiffusionEnergyFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscInt *uOff_x, const PetscScalar *fieldL,
                                                                                      const PetscScalar *fieldR, const PetscScalar *gradL, const PetscScalar *gradR, const PetscInt *aOff,
                                                                                      const PetscInt *aOff_x, const PetscScalar *auxL, const PetscScalar *auxR, const PetscScalar *gradAuxL,
                                                                                      const PetscScalar *gradAuxR, PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int yi = 0;
    const int densityYi = 1;
    const int euler = 0;

    auto flowParameters = (DiffusionData *)ctx;

    // get the current density from euler
    const PetscReal density = 0.5 * (fieldL[uOff[euler] + RHO] + fieldR[uOff[euler] + RHO]);

    // compute the temperature in this volume
    PetscErrorCode ierr;
    PetscReal temperatureLeft;
    ierr = flowParameters->computeTemperatureFunction(dim,
                                                      fieldL[uOff[euler] + RHO],
                                                      fieldL[uOff[euler] + RHOE] / fieldL[uOff[euler] + RHO],
                                                      fieldL + uOff[euler] + RHOU,
                                                      fieldL + uOff[densityYi],
                                                      &temperatureLeft,
                                                      flowParameters->computeTemperatureContext);
    CHKERRQ(ierr);

    PetscReal temperatureRight;
    ierr = flowParameters->computeTemperatureFunction(dim,
                                                      fieldR[uOff[euler] + RHO],
                                                      fieldR[uOff[euler] + RHOE] / fieldR[uOff[euler] + RHO],
                                                      fieldR + uOff[euler] + RHOU,
                                                      fieldR + uOff[densityYi],
                                                      &temperatureRight,
                                                      flowParameters->computeTemperatureContext);
    CHKERRQ(ierr);

    // compute the enthalpy for each species
    PetscReal temperature = 0.5 * (temperatureLeft + temperatureRight);
    flowParameters->computeSpeciesSensibleEnthalpyFunction(temperature, &flowParameters->speciesSpeciesSensibleEnthalpy[0], flowParameters->computeSpeciesSensibleEnthalpyContext);

    // set the non rho E fluxes to zero
    flux[RHO] = 0.0;
    flux[RHOE] = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        flux[RHOU + d] = 0.0;
    }

    // compute diff
    PetscReal diffLeft = 0.0;
    flowParameters->diffFunction(temperatureLeft, fieldL[uOff[euler] + RHO], auxL + aOff[yi], diffLeft, flowParameters->diffContext);
    PetscReal diffRight = 0.0;
    flowParameters->diffFunction(temperatureRight, fieldR[uOff[euler] + RHO], auxR + aOff[yi], diffRight, flowParameters->diffContext);
    PetscReal diff = 0.5 * (diffLeft + diffRight);

    for (PetscInt sp = 0; sp < flowParameters->numberSpecies; ++sp) {
        for (PetscInt d = 0; d < dim; ++d) {
            // speciesFlux(-rho Di dYi/dx - rho Di dYi/dy - rho Di dYi//dz) . n A
            const int offset = aOff_x[yi] + (sp * dim) + d;
            PetscReal speciesFlux = -fg->normal[d] * density * diff * flowParameters->speciesSpeciesSensibleEnthalpy[sp] * 0.5 * (gradAuxL[offset] + gradAuxR[offset]);
            flux[RHOE] += speciesFlux;
        }
    }

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::finiteVolume::processes::SpeciesTransport::DiffusionSpeciesFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscInt *uOff_x, const PetscScalar *fieldL,
                                                                                       const PetscScalar *fieldR, const PetscScalar *gradL, const PetscScalar *gradR, const PetscInt *aOff,
                                                                                       const PetscInt *aOff_x, const PetscScalar *auxL, const PetscScalar *auxR, const PetscScalar *gradAuxL,
                                                                                       const PetscScalar *gradAuxR, PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int yi = 0;
    const int densityYi = 1;
    const int euler = 0;

    auto flowParameters = (DiffusionData *)ctx;

    // get the current density from euler
    const PetscReal density = 0.5 * (fieldL[uOff[euler] + RHO] + fieldR[uOff[euler] + RHO]);

    PetscErrorCode ierr;
    PetscReal temperatureLeft;
    ierr = flowParameters->computeTemperatureFunction(dim,
                                                      fieldL[uOff[euler] + RHO],
                                                      fieldL[uOff[euler] + RHOE] / fieldL[uOff[euler] + RHO],
                                                      fieldL + uOff[euler] + RHOU,
                                                      fieldL + uOff[densityYi],
                                                      &temperatureLeft,
                                                      flowParameters->computeTemperatureContext);
    CHKERRQ(ierr);

    PetscReal temperatureRight;
    ierr = flowParameters->computeTemperatureFunction(dim,
                                                      fieldR[uOff[euler] + RHO],
                                                      fieldR[uOff[euler] + RHOE] / fieldR[uOff[euler] + RHO],
                                                      fieldR + uOff[euler] + RHOU,
                                                      fieldR + uOff[densityYi],
                                                      &temperatureRight,
                                                      flowParameters->computeTemperatureContext);
    CHKERRQ(ierr);

    // compute diff
    PetscReal diffLeft = 0.0;
    flowParameters->diffFunction(temperatureLeft, fieldL[uOff[euler] + RHO], auxL + aOff[yi], diffLeft, flowParameters->diffContext);
    PetscReal diffRight = 0.0;
    flowParameters->diffFunction(temperatureRight, fieldR[uOff[euler] + RHO], auxR + aOff[yi], diffRight, flowParameters->diffContext);
    PetscReal diff = 0.5 * (diffLeft + diffRight);

    // species equations
    for (PetscInt sp = 0; sp < flowParameters->numberSpecies; ++sp) {
        flux[sp] = 0;
        for (PetscInt d = 0; d < dim; ++d) {
            // speciesFlux(-rho Di dYi/dx - rho Di dYi/dy - rho Di dYi//dz) . n A
            const int offset = aOff_x[yi] + (sp * dim) + d;
            PetscReal speciesFlux = -fg->normal[d] * density * diff * 0.5 * (gradAuxL[offset] + gradAuxR[offset]);
            flux[sp] += speciesFlux;
        }
    }

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::finiteVolume::processes::SpeciesTransport::AdvectionFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscInt *uOff_x, const PetscScalar *fieldL,
                                                                                const PetscScalar *fieldR, const PetscScalar *gradL, const PetscScalar *gradR, const PetscInt *aOff,
                                                                                const PetscInt *aOff_x, const PetscScalar *auxL, const PetscScalar *auxR, const PetscScalar *gradAuxL,
                                                                                const PetscScalar *gradAuxR, PetscScalar *flux, void *ctx) {
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
    PetscReal ML;
    PetscReal pL;
    DecodeEulerState(eulerAdvectionData->decodeStateFunction,
                     eulerAdvectionData->decodeStateContext,
                     dim,
                     fieldL + uOff[EULER_FIELD],
                     fieldL + uOff[YI_FIELD],
                     norm,
                     &densityL,
                     &normalVelocityL,
                     velocityL,
                     &internalEnergyL,
                     &aL,
                     &ML,
                     &pL);

    PetscReal densityR;
    PetscReal normalVelocityR;
    PetscReal velocityR[3];
    PetscReal internalEnergyR;
    PetscReal aR;
    PetscReal MR;
    PetscReal pR;
    DecodeEulerState(eulerAdvectionData->decodeStateFunction,
                     eulerAdvectionData->decodeStateContext,
                     dim,
                     fieldR + uOff[EULER_FIELD],
                     fieldR + uOff[YI_FIELD],
                     norm,
                     &densityR,
                     &normalVelocityR,
                     velocityR,
                     &internalEnergyR,
                     &aR,
                     &MR,
                     &pR);

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
    IS cellIS;
    PetscInt cStart, cEnd;
    const PetscInt *cells;
    solver.GetCellRange(cellIS, cStart, cEnd, cells);

    for (PetscInt c = cStart; c < cEnd; ++c) {
        PetscInt cell = cells ? cells[c] : c;

        // Get the euler and density field
        const PetscScalar *euler = nullptr;
        DMPlexPointGlobalFieldRead(dm, cell, eulerFieldInfo.id, solutionArray, &euler) >> checkError;
        PetscScalar *densityYi;
        DMPlexPointGlobalFieldRef(dm, cell, densityYiFieldInfo.id, solutionArray, &densityYi) >> checkError;

        // Only update if in the global vector
        if (euler) {
            // Get density
            const PetscScalar density = euler[RHO];

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
    solver.RestoreRange(cellIS, cStart, cEnd, cells);
};

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::SpeciesTransport, "diffusion/advection for the species yi field",
         ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"),
         OPT(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculator", "the flux calculator (default is no advection)"),
         OPT(ablate::eos::transport::TransportModel, "transport", "the diffusion transport model (default is no diffusion)"));
