#include "evTransport.hpp"
#include <utilities/mathUtilities.hpp>
#include "finiteVolume/compressibleFlowFields.hpp"

ablate::finiteVolume::processes::EVTransport::EVTransport(std::string conserved, std::string nonConserved, std::shared_ptr<eos::EOS> eosIn, std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalcIn,
                                                          std::shared_ptr<eos::transport::TransportModel> transportModelIn, std::shared_ptr<resources::PressureGradientScaling> pressureGradientScaling)
    : FlowProcess(pressureGradientScaling),
      conserved(std::move(conserved)),
      nonConserved(std::move(nonConserved)),
      fluxCalculator(std::move(fluxCalcIn)),
      eos(std::move(eosIn)),
      transportModel(std::move(transportModelIn)),
      advectionData() {
    if (fluxCalculator) {
        // set the decode state function
        advectionData.decodeStateFunction = eos->GetDecodeStateFunction();
        advectionData.decodeStateContext = eos->GetDecodeStateContext();
        advectionData.numberEV = 0;

        // extract the difference function from fluxDifferencer object
        advectionData.fluxCalculatorFunction = fluxCalculator->GetFluxCalculatorFunction();
        advectionData.fluxCalculatorCtx = fluxCalculator->GetFluxCalculatorContext();

        if (pressureGradientScaling) {
            advectionData.pgsAlpha = &pressureGradientScaling->GetAlpha();
        }
    }

    if (transportModel) {
        diffusionData.diffFunction = transportModel->GetComputeDiffusivityFunction();
        diffusionData.diffContext = transportModel->GetComputeDiffusivityContext();

        // set the eos functions
        diffusionData.numberEV = 0;
        diffusionData.computeTemperatureFunction = eos->GetComputeTemperatureFunction();
        diffusionData.computeTemperatureContext = eos->GetComputeTemperatureContext();

        diffusionData.speciesSpeciesSensibleEnthalpy.resize(eos->GetSpecies().size());
    } else {
        diffusionData.diffFunction = nullptr;
        diffusionData.diffContext = nullptr;
    }

    numberEV = 0;
}

void ablate::finiteVolume::processes::EVTransport::Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    ablate::finiteVolume::processes::FlowProcess::Initialize(flow);

    if (flow.GetSubDomain().ContainsField(conserved)) {
        // determine the number of components in the ev
        auto conservedForm = flow.GetSubDomain().GetField(conserved);
        advectionData.numberEV = conservedForm.numberComponents;
        numberEV = conservedForm.numberComponents;
        diffusionData.numberEV = conservedForm.numberComponents;
        if (!flow.GetSubDomain().ContainsField(nonConserved)) {
            throw std::invalid_argument("The ablate::finiteVolume::processes::EVTransport process expects the conserved (" + conserved + ") and non-conserved (" + nonConserved +
                                        ") extra variables to be in the flow.");
        }

        if (fluxCalculator) {
            if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::DENSITY_YI_FIELD)) {
                flow.RegisterRHSFunction(AdvectionFlux, &advectionData, conserved, {CompressibleFlowFields::EULER_FIELD, conserved, CompressibleFlowFields::DENSITY_YI_FIELD}, {});
            } else {
                flow.RegisterRHSFunction(AdvectionFlux, &advectionData, conserved, {CompressibleFlowFields::EULER_FIELD, conserved}, {});
            }
        }

        if (diffusionData.diffFunction) {
            if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::YI_FIELD)) {
                flow.RegisterRHSFunction(DiffusionEVFlux, &diffusionData, conserved, {CompressibleFlowFields::EULER_FIELD}, {nonConserved, CompressibleFlowFields::YI_FIELD});
            } else {
                flow.RegisterRHSFunction(DiffusionEVFlux, &diffusionData, conserved, {CompressibleFlowFields::EULER_FIELD}, {nonConserved});
            }
        }

        flow.RegisterAuxFieldUpdate(UpdateEVField, &numberEV, nonConserved, {CompressibleFlowFields::EULER_FIELD, conserved});
    }
}

PetscErrorCode ablate::finiteVolume::processes::EVTransport::UpdateEVField(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt *uOff, const PetscScalar *conservedValues,
                                                                           PetscScalar *auxField, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[uOff[0] + RHO];

    auto numberEV = (PetscInt *)ctx;

    for (PetscInt e = 0; e < *numberEV; e++) {
        auxField[e] = conservedValues[uOff[1] + e] / density;
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::EVTransport::AdvectionFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscInt *uOff_x, const PetscScalar *fieldL,
                                                                           const PetscScalar *fieldR, const PetscScalar *gradL, const PetscScalar *gradR, const PetscInt *aOff, const PetscInt *aOff_x,
                                                                           const PetscScalar *auxL, const PetscScalar *auxR, const PetscScalar *gradAuxL, const PetscScalar *gradAuxR,
                                                                           PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
    auto eulerAdvectionData = (AdvectionData *)ctx;

    // Compute the norm
    PetscReal norm[3];
    utilities::MathUtilities::NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = utilities::MathUtilities::MagVector(dim, fg->normal);

    const int EULER_FIELD = 0;
    const int DENSITY_EV_FIELD = 1;
    const int DENSITY_YI_FIELD = 2;

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
                     fieldL + uOff[DENSITY_YI_FIELD],
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
                     fieldR + uOff[DENSITY_YI_FIELD],
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

    // Update the speed of sound if pgs is provided
    if (eulerAdvectionData->pgsAlpha) {
        aL /= *eulerAdvectionData->pgsAlpha;
        aR /= *eulerAdvectionData->pgsAlpha;
    }

    if (eulerAdvectionData->fluxCalculatorFunction(eulerAdvectionData->fluxCalculatorCtx, normalVelocityL, aL, densityL, pL, normalVelocityR, aR, densityR, pR, &massFlux, NULL) ==
        fluxCalculator::LEFT) {
        // march over each gas species
        for (PetscInt ev = 0; ev < eulerAdvectionData->numberEV; ev++) {
            // Note: there is no density in the flux because uR and UL are density*yi
            flux[ev] = (massFlux * fieldL[uOff[DENSITY_EV_FIELD] + ev] / densityL) * areaMag;
        }
    } else {
        // march over each gas species
        for (PetscInt ev = 0; ev < eulerAdvectionData->numberEV; ev++) {
            // Note: there is no density in the flux because uR and UL are density*yi
            flux[ev] = (massFlux * fieldR[uOff[DENSITY_EV_FIELD] + ev] / densityR) * areaMag;
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::EVTransport::DiffusionEVFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscInt *uOff_x, const PetscScalar *fieldL,
                                                                             const PetscScalar *fieldR, const PetscScalar *gradL, const PetscScalar *gradR, const PetscInt *aOff,
                                                                             const PetscInt *aOff_x, const PetscScalar *auxL, const PetscScalar *auxR, const PetscScalar *gradAuxL,
                                                                             const PetscScalar *gradAuxR, PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int EULER_FIELD = 0;
    const int YI_FIELD = 1;
    const int EV_FIELD = 0;

    auto flowParameters = (DiffusionData *)ctx;

    // get the current density from euler
    const PetscReal density = 0.5 * (fieldL[uOff[EULER_FIELD] + RHO] + fieldR[uOff[EULER_FIELD] + RHO]);

    PetscErrorCode ierr;
    PetscReal temperatureLeft;
    ierr = flowParameters->computeTemperatureFunction(dim,
                                                      fieldL[uOff[EULER_FIELD] + RHO],
                                                      fieldL[uOff[EULER_FIELD] + RHOE] / fieldL[uOff[EULER_FIELD] + RHO],
                                                      fieldL + uOff[EULER_FIELD] + RHOU,
                                                      auxL + aOff[YI_FIELD],
                                                      &temperatureLeft,
                                                      flowParameters->computeTemperatureContext);
    CHKERRQ(ierr);

    PetscReal temperatureRight;
    ierr = flowParameters->computeTemperatureFunction(dim,
                                                      fieldR[uOff[EULER_FIELD] + RHO],
                                                      fieldR[uOff[EULER_FIELD] + RHOE] / fieldR[uOff[EULER_FIELD] + RHO],
                                                      fieldR + uOff[EULER_FIELD] + RHOU,
                                                      auxR + aOff[YI_FIELD],
                                                      &temperatureRight,
                                                      flowParameters->computeTemperatureContext);
    CHKERRQ(ierr);

    // compute diff
    PetscReal diffLeft = 0.0;
    flowParameters->diffFunction(temperatureLeft, fieldL[uOff[EULER_FIELD] + RHO], auxL + aOff[YI_FIELD], diffLeft, flowParameters->diffContext);
    PetscReal diffRight = 0.0;
    flowParameters->diffFunction(temperatureRight, fieldR[uOff[EULER_FIELD] + RHO], auxR + aOff[YI_FIELD], diffRight, flowParameters->diffContext);
    PetscReal diff = 0.5 * (diffLeft + diffRight);

    // species equations
    for (PetscInt ev = 0; ev < flowParameters->numberEV; ++ev) {
        flux[ev] = 0;
        for (PetscInt d = 0; d < dim; ++d) {
            // speciesFlux(-rho Di dYi/dx - rho Di dYi/dy - rho Di dYi//dz) . n A
            const int offset = aOff_x[EV_FIELD] + (ev * dim) + d;
            PetscReal evFlux = -fg->normal[d] * density * diff * 0.5 * (gradAuxL[offset] + gradAuxR[offset]);
            flux[ev] += evFlux;
        }
    }

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::EVTransport, "diffusion/advection for the specified EV",
         ARG(std::string, "conserved", "the name of the conserved (density*ev) of the variable"), ARG(std::string, "nonConserved", "the name of the non-conserved (ev) of the variable"),
         ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"),
         OPT(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculator", "the flux calculator (default is no advection)"),
         OPT(ablate::eos::transport::TransportModel, "transport", "the diffusion transport model (default is no diffusion)"),
         OPT(ablate::finiteVolume::resources::PressureGradientScaling, "pgs", "Pressure gradient scaling is used to scale the acoustic propagation speed and increase time step for low speed flows"));
