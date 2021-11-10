#include "eulerTransport.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/fluxCalculator/ausm.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscError.hpp"

ablate::finiteVolume::processes::EulerTransport::EulerTransport(std::shared_ptr<parameters::Parameters> parametersIn, std::shared_ptr<eos::EOS> eosIn,
                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorIn, std::shared_ptr<eos::transport::TransportModel> transportModelIn)
    : fluxCalculator(std::move(fluxCalculatorIn)), eos(std::move(eosIn)), transportModel(std::move(transportModelIn)), advectionData(), updateTemperatureData() {
    // If there is a flux calculator assumed advection
    if (fluxCalculator) {
        // cfl
        advectionData.cfl = parametersIn->Get<PetscReal>("cfl", 0.5);

        // set the decode state function
        advectionData.decodeStateFunction = eos->GetDecodeStateFunction();
        advectionData.decodeStateContext = eos->GetDecodeStateContext();
        advectionData.numberSpecies = (PetscInt)eos->GetSpecies().size();

        // extract the difference function from fluxDifferencer object
        advectionData.fluxCalculatorFunction = fluxCalculator->GetFluxCalculatorFunction();
        advectionData.fluxCalculatorCtx = fluxCalculator->GetFluxCalculatorContext();
    }

    // If there is a transport model, assumed diffusion
    if (transportModel) {
        // Store the required data for the low level c functions
        diffusionData.muFunction = transportModel->GetComputeViscosityFunction();
        diffusionData.muContext = transportModel->GetComputeViscosityContext();
        diffusionData.kFunction = transportModel->GetComputeConductivityFunction();
        diffusionData.kContext = transportModel->GetComputeConductivityContext();

    } else {
        // Store the required data for the low level c functions
        diffusionData.muFunction = nullptr;
        diffusionData.muContext = nullptr;
        diffusionData.kFunction = nullptr;
        diffusionData.kContext = nullptr;
    }

    // set the decode state function
    diffusionData.numberSpecies = (PetscInt)eos->GetSpecies().size();
    diffusionData.yiScratch.resize(eos->GetSpecies().size());

    updateTemperatureData.computeTemperatureFunction = eos->GetComputeTemperatureFunction();
    updateTemperatureData.computeTemperatureContext = eos->GetComputeTemperatureContext();
    updateTemperatureData.numberSpecies = (PetscInt)eos->GetSpecies().size();
}

void ablate::finiteVolume::processes::EulerTransport::Initialize(ablate::finiteVolume::FiniteVolumeSolver& flow) {
    // Register the euler source terms
    if (fluxCalculator) {
        if (eos->GetSpecies().empty()) {
            flow.RegisterRHSFunction(AdvectionFlux, &advectionData, CompressibleFlowFields::EULER_FIELD, {CompressibleFlowFields::EULER_FIELD}, {});
        } else {
            flow.RegisterRHSFunction(AdvectionFlux, &advectionData, CompressibleFlowFields::EULER_FIELD, {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD}, {});
        }

        // PetscErrorCode PetscOptionsGetBool(PetscOptions options,const char pre[],const char name[],PetscBool *ivalue,PetscBool *set)
        PetscBool automaticTimeStepCalculator = PETSC_TRUE;
        PetscOptionsGetBool(nullptr, nullptr, "-automaticTimeStepCalculator", &automaticTimeStepCalculator, nullptr);
        if (automaticTimeStepCalculator) {
            flow.RegisterComputeTimeStepFunction(ComputeTimeStep, &advectionData);
        }
    }

    // if there are any coefficients for diffusion, compute diffusion
    if (diffusionData.kFunction || diffusionData.muFunction) {
        // Register the euler diffusion source terms
        if (diffusionData.numberSpecies > 0) {
            flow.RegisterRHSFunction(DiffusionFlux,
                                     &diffusionData,
                                     CompressibleFlowFields::EULER_FIELD,
                                     {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD},
                                     {CompressibleFlowFields::TEMPERATURE_FIELD, CompressibleFlowFields::VELOCITY_FIELD});
        } else {
            flow.RegisterRHSFunction(DiffusionFlux,
                                     &diffusionData,
                                     CompressibleFlowFields::EULER_FIELD,
                                     {CompressibleFlowFields::EULER_FIELD},
                                     {CompressibleFlowFields::TEMPERATURE_FIELD, CompressibleFlowFields::VELOCITY_FIELD});
        }
    }

    // check to see if auxFieldUpdates needed to be added
    if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::VELOCITY_FIELD)) {
        flow.RegisterAuxFieldUpdate(UpdateAuxVelocityField, nullptr, CompressibleFlowFields::VELOCITY_FIELD, {CompressibleFlowFields::EULER_FIELD});
    }
    if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::TEMPERATURE_FIELD)) {
        if (diffusionData.numberSpecies > 0) {
            // add in aux update variables
            flow.RegisterAuxFieldUpdate(
                UpdateAuxTemperatureField, &updateTemperatureData, CompressibleFlowFields::TEMPERATURE_FIELD, {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD});
        } else {
            // add in aux update variables
            flow.RegisterAuxFieldUpdate(UpdateAuxTemperatureField, &updateTemperatureData, CompressibleFlowFields::TEMPERATURE_FIELD, {CompressibleFlowFields::EULER_FIELD});
        }
    }
}

PetscErrorCode ablate::finiteVolume::processes::EulerTransport::AdvectionFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt* uOff, const PetscInt* uOff_x, const PetscScalar* fieldL,
                                                                              const PetscScalar* fieldR, const PetscScalar* gradL, const PetscScalar* gradR, const PetscInt* aOff,
                                                                              const PetscInt* aOff_x, const PetscScalar* auxL, const PetscScalar* auxR, const PetscScalar* gradAuxL,
                                                                              const PetscScalar* gradAuxR, PetscScalar* flux, void* ctx) {
    PetscFunctionBeginUser;

    auto eulerAdvectionData = (AdvectionData*)ctx;

    const int EULER_FIELD = 0;
    const int YI_FIELD = 1;

    // Compute the norm
    PetscReal norm[3];
    utilities::MathUtilities::NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = utilities::MathUtilities::MagVector(dim, fg->normal);

    // Decode the left and right states
    PetscReal densityL;
    PetscReal normalVelocityL;
    PetscReal velocityL[3];
    PetscReal internalEnergyL;
    PetscReal aL;
    PetscReal ML;
    PetscReal pL;

    const PetscReal* densityYiL = eulerAdvectionData->numberSpecies > 0 ? fieldL + uOff[YI_FIELD] : nullptr;
    DecodeEulerState(eulerAdvectionData->decodeStateFunction,
                     eulerAdvectionData->decodeStateContext,
                     dim,
                     fieldL + uOff[EULER_FIELD],
                     densityYiL,
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
    const PetscReal* densityYiR = eulerAdvectionData->numberSpecies > 0 ? fieldR + uOff[YI_FIELD] : nullptr;
    DecodeEulerState(eulerAdvectionData->decodeStateFunction,
                     eulerAdvectionData->decodeStateContext,
                     dim,
                     fieldR + uOff[EULER_FIELD],
                     densityYiR,
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
    PetscReal p12;

    /*void (*)(void* ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL,
        PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR,
        PetscReal * m12, PetscReal *p12);*/
    fluxCalculator::Direction direction =
        eulerAdvectionData->fluxCalculatorFunction(eulerAdvectionData->fluxCalculatorCtx, normalVelocityL, aL, densityL, pL, normalVelocityR, aR, densityR, pR, &massFlux, &p12);

    if (direction == fluxCalculator::LEFT) {
        flux[RHO] = massFlux * areaMag;
        PetscReal velMagL = utilities::MathUtilities::MagVector(dim, velocityL);
        PetscReal HL = internalEnergyL + velMagL * velMagL / 2.0 + pL / densityL;
        flux[RHOE] = HL * massFlux * areaMag;
        for (PetscInt n = 0; n < dim; n++) {
            flux[RHOU + n] = velocityL[n] * massFlux * areaMag + p12 * fg->normal[n];
        }
    } else if (direction == fluxCalculator::RIGHT) {
        flux[RHO] = massFlux * areaMag;
        PetscReal velMagR = utilities::MathUtilities::MagVector(dim, velocityR);
        PetscReal HR = internalEnergyR + velMagR * velMagR / 2.0 + pR / densityR;
        flux[RHOE] = HR * massFlux * areaMag;
        for (PetscInt n = 0; n < dim; n++) {
            flux[RHOU + n] = velocityR[n] * massFlux * areaMag + p12 * fg->normal[n];
        }
    } else {
        flux[RHO] = massFlux * areaMag;

        PetscReal velMagL = utilities::MathUtilities::MagVector(dim, velocityL);
        PetscReal HL = internalEnergyL + velMagL * velMagL / 2.0 + pL / densityL;

        PetscReal velMagR = utilities::MathUtilities::MagVector(dim, velocityR);
        PetscReal HR = internalEnergyR + velMagR * velMagR / 2.0 + pR / densityR;

        flux[RHOE] = 0.5 * (HL + HR) * massFlux * areaMag;
        for (PetscInt n = 0; n < dim; n++) {
            flux[RHOU + n] = 0.5 * (velocityL[n] + velocityR[n]) * massFlux * areaMag + p12 * fg->normal[n];
        }
    }

    PetscFunctionReturn(0);
}

double ablate::finiteVolume::processes::EulerTransport::ComputeTimeStep(TS ts, ablate::finiteVolume::FiniteVolumeSolver& flow, void* ctx) {
    // Get the dm and current solution vector
    DM dm;
    TSGetDM(ts, &dm) >> checkError;
    Vec v;
    TSGetSolution(ts, &v) >> checkError;

    // Get the flow param
    auto advectionData = (AdvectionData*)ctx;

    // Get the fv geom
    PetscReal minCellRadius;
    DMPlexGetGeometryFVM(dm, NULL, NULL, &minCellRadius) >> checkError;

    // Get the valid cell range over this region
    IS cellIS;
    PetscInt cStart, cEnd;
    const PetscInt* cells;
    flow.GetCellRange(cellIS, cStart, cEnd, cells);

    const PetscScalar* x;
    VecGetArrayRead(v, &x) >> checkError;

    // Get the dim from the dm
    PetscInt dim;
    DMGetDimension(dm, &dim) >> checkError;

    // assume the smallest cell is the limiting factor for now
    const PetscReal dx = 2.0 * minCellRadius;

    // Get field location for euler and densityYi
    auto eulerId = flow.GetSubDomain().GetField("euler").id;
    auto densityYiId = advectionData->numberSpecies > 0 ? flow.GetSubDomain().GetField("densityYi").id : -1;

    // March over each cell
    PetscReal dtMin = 1000.0;
    for (PetscInt c = cStart; c < cEnd; ++c) {
        PetscInt cell = cells ? cells[c] : c;

        const PetscReal* xc;
        const PetscReal* densityYi = NULL;
        DMPlexPointGlobalFieldRead(dm, cell, eulerId, x, &xc) >> checkError;

        if (densityYiId >= 0) {
            DMPlexPointGlobalFieldRead(dm, cell, densityYiId, x, &densityYi) >> checkError;
        }

        if (xc) {  // must be real cell and not ghost
            PetscReal rho = xc[RHO];
            PetscReal vel[3];
            for (PetscInt i = 0; i < dim; i++) {
                vel[i] = xc[RHOU + i] / rho;
            }

            // Get the speed of sound from the eos
            PetscReal ie;
            PetscReal a;
            PetscReal p;
            advectionData->decodeStateFunction(dim, rho, xc[RHOE] / rho, vel, densityYi, &ie, &a, &p, advectionData->decodeStateContext) >> checkError;

            PetscReal u = xc[RHOU] / rho;
            PetscReal dt = advectionData->cfl * dx / (a + PetscAbsReal(u));

            dtMin = PetscMin(dtMin, dt);
        }
    }
    VecRestoreArrayRead(v, &x) >> checkError;
    flow.RestoreRange(cellIS, cStart, cEnd, cells);
    return dtMin;
}
PetscErrorCode ablate::finiteVolume::processes::EulerTransport::DiffusionFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt* uOff, const PetscInt* uOff_x, const PetscScalar* fieldL,
                                                                              const PetscScalar* fieldR, const PetscScalar* gradL, const PetscScalar* gradR, const PetscInt* aOff,
                                                                              const PetscInt* aOff_x, const PetscScalar* auxL, const PetscScalar* auxR, const PetscScalar* gradAuxL,
                                                                              const PetscScalar* gradAuxR, PetscScalar* flux, void* ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int T = 0;
    const int VEL = 1;
    const int EULER = 0;
    const int DENSITY_YI = 1;

    PetscErrorCode ierr;
    auto flowParameters = (DiffusionData*)ctx;

    // Compute mu and k
    PetscReal* yiScratch = &flowParameters->yiScratch[0];
    for (std::size_t s = 0; s < flowParameters->yiScratch.size(); s++) {
        yiScratch[s] = fieldL[uOff[DENSITY_YI] + s] / fieldL[uOff[EULER] + RHO];
    }

    PetscReal muLeft = 0.0;
    flowParameters->muFunction(auxL[aOff[T]], fieldL[uOff[EULER] + RHO], yiScratch, muLeft, flowParameters->muContext);
    PetscReal kLeft = 0.0;
    flowParameters->kFunction(auxL[aOff[T]], fieldL[uOff[EULER] + RHO], yiScratch, kLeft, flowParameters->kContext);

    // Compute mu and k
    for (std::size_t s = 0; s < flowParameters->yiScratch.size(); s++) {
        yiScratch[s] = fieldR[uOff[DENSITY_YI] + s] / fieldR[uOff[EULER] + RHO];
    }

    PetscReal muRight = 0.0;
    flowParameters->muFunction(auxR[aOff[T]], fieldR[uOff[EULER] + RHO], yiScratch, muRight, flowParameters->muContext);
    PetscReal kRight = 0.0;
    flowParameters->kFunction(auxR[aOff[T]], fieldR[uOff[EULER] + RHO], yiScratch, kRight, flowParameters->kContext);

    // Compute the stress tensor tau
    PetscReal tau[9];  // Maximum size without symmetry
    ierr = CompressibleFlowComputeStressTensor(dim, 0.5 * (muLeft + muRight), gradAuxL + aOff_x[VEL], gradAuxR + aOff_x[VEL], tau);
    CHKERRQ(ierr);

    // for each velocity component
    for (PetscInt c = 0; c < dim; ++c) {
        PetscReal viscousFlux = 0.0;

        // March over each direction
        for (PetscInt d = 0; d < dim; ++d) {
            viscousFlux += -fg->normal[d] * tau[c * dim + d];  // This is tau[c][d]
        }

        // add in the contribution
        flux[RHOU + c] = viscousFlux;
    }

    // energy equation
    flux[RHOE] = 0.0;
    for (PetscInt d = 0; d < dim; ++d) {
        PetscReal heatFlux = 0.0;
        // add in the contributions for this viscous terms
        for (PetscInt c = 0; c < dim; ++c) {
            heatFlux += 0.5 * (auxL[aOff[VEL] + c] + auxR[aOff[VEL] + c]) * tau[d * dim + c];
        }

        // heat conduction (-k dT/dx - k dT/dy - k dT/dz) . n A
        heatFlux += 0.5 * (kLeft * gradAuxL[aOff_x[T] + d] + kRight * gradAuxR[aOff_x[T] + d]);

        // Multiply by the area normal
        heatFlux *= -fg->normal[d];

        flux[RHOE] += heatFlux;
    }

    // zero out the density flux
    flux[RHO] = 0.0;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::EulerTransport::CompressibleFlowComputeStressTensor(PetscInt dim, PetscReal mu, const PetscReal* gradVelL, const PetscReal* gradVelR, PetscReal* tau) {
    PetscFunctionBeginUser;
    // pre compute the div of the velocity field
    PetscReal divVel = 0.0;
    for (PetscInt c = 0; c < dim; ++c) {
        divVel += 0.5 * (gradVelL[c * dim + c] + gradVelR[c * dim + c]);
    }

    // March over each velocity component, u, v, w
    for (PetscInt c = 0; c < dim; ++c) {
        // March over each physical coordinates
        for (PetscInt d = 0; d < dim; ++d) {
            if (d == c) {
                // for the xx, yy, zz, components
                tau[c * dim + d] = 2.0 * mu * (0.5 * (gradVelL[c * dim + d] + gradVelR[c * dim + d]) - divVel / 3.0);
            } else {
                // for xy, xz, etc
                tau[c * dim + d] = mu * (0.5 * (gradVelL[c * dim + d] + gradVelR[c * dim + d]) + 0.5 * (gradVelL[d * dim + c] + gradVelR[d * dim + c]));
            }
        }
    }
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::EulerTransport::UpdateAuxVelocityField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[],
                                                                                       const PetscScalar* conservedValues, PetscScalar* auxField, void* ctx) {
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[uOff[0] + RHO];

    for (PetscInt d = 0; d < dim; d++) {
        auxField[d] = conservedValues[uOff[0] + RHOU + d] / density;
    }

    PetscFunctionReturn(0);
}

// When used, you must request euler, then densityYi
PetscErrorCode ablate::finiteVolume::processes::EulerTransport::UpdateAuxTemperatureField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[],
                                                                                          const PetscScalar* conservedValues, PetscScalar* auxField, void* ctx) {
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[uOff[0] + RHO];
    PetscReal totalEnergy = conservedValues[uOff[0] + RHOE] / density;
    auto flowParameters = (UpdateTemperatureData*)ctx;
    PetscErrorCode ierr = flowParameters->computeTemperatureFunction(
        dim, density, totalEnergy, conservedValues + uOff[0] + RHOU, flowParameters->numberSpecies ? conservedValues + uOff[1] : NULL, auxField, flowParameters->computeTemperatureContext);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

#include "parser/registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::EulerTransport, "build advection/diffusion for the euler field",
         OPT(ablate::parameters::Parameters, "parameters", "the parameters used by advection"), ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"),
         OPT(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculator", "the flux calculator (default is no advection)"),
         OPT(eos::transport::TransportModel, "transport", "the diffusion transport model (default is no diffusion)"));
