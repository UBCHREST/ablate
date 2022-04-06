#include "eulerTransport.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/fluxCalculator/ausm.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscError.hpp"

ablate::finiteVolume::processes::EulerTransport::EulerTransport(const std::shared_ptr<parameters::Parameters>& parametersIn, std::shared_ptr<eos::EOS> eosIn,
                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorIn, std::shared_ptr<eos::transport::TransportModel> transportModelIn)
    : fluxCalculator(std::move(fluxCalculatorIn)), eos(std::move(eosIn)), transportModel(std::move(transportModelIn)), advectionData(), updateTemperatureData() {
    // If there is a flux calculator assumed advection
    if (fluxCalculator) {
        // cfl
        advectionData.cfl = parametersIn->Get<PetscReal>("cfl", 0.5);

        // extract the difference function from fluxDifferencer object
        advectionData.fluxCalculatorFunction = fluxCalculator->GetFluxCalculatorFunction();
        advectionData.fluxCalculatorCtx = fluxCalculator->GetFluxCalculatorContext();
    }

    advectionData.numberSpecies = (PetscInt)eos->GetSpecies().size();

    // set the decode state function
    diffusionData.numberSpecies = (PetscInt)eos->GetSpecies().size();
    diffusionData.yiScratch.resize(eos->GetSpecies().size());

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
        flow.RegisterComputeTimeStepFunction(ComputeTimeStep, &advectionData, "cfl");

        advectionData.computeTemperature = eos->GetThermodynamicFunction(eos::ThermodynamicProperty::Temperature, flow.GetSubDomain().GetFields());
        advectionData.computeInternalEnergy = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::InternalSensibleEnergy, flow.GetSubDomain().GetFields());
        advectionData.computeSpeedOfSound = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SpeedOfSound, flow.GetSubDomain().GetFields());
        advectionData.computePressure = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Pressure, flow.GetSubDomain().GetFields());
    }

    // if there are any coefficients for diffusion, compute diffusion
    if (transportModel) {
        // Store the required data for the low level c functions
        diffusionData.muFunction = transportModel->GetTransportTemperatureFunction(eos::transport::TransportProperty::Viscosity, flow.GetSubDomain().GetFields());
        diffusionData.kFunction = transportModel->GetTransportTemperatureFunction(eos::transport::TransportProperty::Conductivity, flow.GetSubDomain().GetFields());

        if (diffusionData.muFunction.function || diffusionData.kFunction.function) {
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
    }

    // check to see if auxFieldUpdates needed to be added
    if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::VELOCITY_FIELD)) {
        flow.RegisterAuxFieldUpdate(UpdateAuxVelocityField, nullptr, std::vector<std::string>{CompressibleFlowFields::VELOCITY_FIELD}, {CompressibleFlowFields::EULER_FIELD});
    }
    if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::TEMPERATURE_FIELD)) {
        if (updateTemperatureData.numberSpecies > 0) {
            // add in aux update variables
            flow.RegisterAuxFieldUpdate(UpdateAuxTemperatureField,
                                        &updateTemperatureData,
                                        std::vector<std::string>{CompressibleFlowFields::TEMPERATURE_FIELD},
                                        {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD});
        } else {
            // add in aux update variables
            flow.RegisterAuxFieldUpdate(UpdateAuxTemperatureField, &updateTemperatureData, std::vector<std::string>{CompressibleFlowFields::TEMPERATURE_FIELD}, {CompressibleFlowFields::EULER_FIELD});
        }
        // set decode state functions
        updateTemperatureData.computeTemperatureFunction = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Temperature, flow.GetSubDomain().GetFields());
    }

    if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::PRESSURE_FIELD)) {
        if (advectionData.numberSpecies > 0) {
            // add in aux update variables
            flow.RegisterAuxFieldUpdate(UpdateAuxPressureField,
                                        &advectionData,
                                        std::vector<std::string>{CompressibleFlowFields::PRESSURE_FIELD},
                                        {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD});
        } else {
            // add in aux update variables
            flow.RegisterAuxFieldUpdate(UpdateAuxPressureField, &advectionData, std::vector<std::string>{CompressibleFlowFields::PRESSURE_FIELD}, {CompressibleFlowFields::EULER_FIELD});
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
    PetscReal p12;

    fluxCalculator::Direction direction =
        eulerAdvectionData->fluxCalculatorFunction(eulerAdvectionData->fluxCalculatorCtx, normalVelocityL, aL, densityL, pL, normalVelocityR, aR, densityR, pR, &massFlux, &p12);

    if (direction == fluxCalculator::LEFT) {
        flux[CompressibleFlowFields::RHO] = massFlux * areaMag;
        PetscReal velMagL = utilities::MathUtilities::MagVector(dim, velocityL);
        PetscReal HL = internalEnergyL + velMagL * velMagL / 2.0 + pL / densityL;
        flux[CompressibleFlowFields::RHOE] = HL * massFlux * areaMag;
        for (PetscInt n = 0; n < dim; n++) {
            flux[CompressibleFlowFields::RHOU + n] = velocityL[n] * massFlux * areaMag + p12 * fg->normal[n];
        }
    } else if (direction == fluxCalculator::RIGHT) {
        flux[CompressibleFlowFields::RHO] = massFlux * areaMag;
        PetscReal velMagR = utilities::MathUtilities::MagVector(dim, velocityR);
        PetscReal HR = internalEnergyR + velMagR * velMagR / 2.0 + pR / densityR;
        flux[CompressibleFlowFields::RHOE] = HR * massFlux * areaMag;
        for (PetscInt n = 0; n < dim; n++) {
            flux[CompressibleFlowFields::RHOU + n] = velocityR[n] * massFlux * areaMag + p12 * fg->normal[n];
        }
    } else {
        flux[CompressibleFlowFields::RHO] = massFlux * areaMag;

        PetscReal velMagL = utilities::MathUtilities::MagVector(dim, velocityL);
        PetscReal HL = internalEnergyL + velMagL * velMagL / 2.0 + pL / densityL;

        PetscReal velMagR = utilities::MathUtilities::MagVector(dim, velocityR);
        PetscReal HR = internalEnergyR + velMagR * velMagR / 2.0 + pR / densityR;

        flux[CompressibleFlowFields::RHOE] = 0.5 * (HL + HR) * massFlux * areaMag;
        for (PetscInt n = 0; n < dim; n++) {
            flux[CompressibleFlowFields::RHOU + n] = 0.5 * (velocityL[n] + velocityR[n]) * massFlux * areaMag + p12 * fg->normal[n];
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

    // March over each cell
    PetscReal dtMin = 1000.0;
    for (PetscInt c = cStart; c < cEnd; ++c) {
        PetscInt cell = cells ? cells[c] : c;

        const PetscReal* euler;
        const PetscReal* conserved = NULL;
        DMPlexPointGlobalFieldRead(dm, cell, eulerId, x, &euler) >> checkError;
        DMPlexPointGlobalRead(dm, cell, x, &conserved) >> checkError;

        if (euler) {  // must be real cell and not ghost
            PetscReal rho = euler[CompressibleFlowFields::RHO];

            // Get the speed of sound from the eos
            PetscReal temperature;
            advectionData->computeTemperature.function(conserved, &temperature, advectionData->computeTemperature.context.get()) >> checkError;
            PetscReal a;
            advectionData->computeSpeedOfSound.function(conserved, temperature, &a, advectionData->computeSpeedOfSound.context.get()) >> checkError;

            PetscReal u = euler[CompressibleFlowFields::RHOU] / rho;
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
        yiScratch[s] = fieldL[uOff[DENSITY_YI] + s] / fieldL[uOff[EULER] + CompressibleFlowFields::RHO];
    }

    PetscReal muLeft = 0.0;
    flowParameters->muFunction.function(fieldL, auxL[aOff[T]], &muLeft, flowParameters->muFunction.context.get());
    PetscReal kLeft = 0.0;
    flowParameters->kFunction.function(fieldL, auxL[aOff[T]], &kLeft, flowParameters->kFunction.context.get());

    // Compute mu and k
    for (std::size_t s = 0; s < flowParameters->yiScratch.size(); s++) {
        yiScratch[s] = fieldR[uOff[DENSITY_YI] + s] / fieldR[uOff[EULER] + CompressibleFlowFields::RHO];
    }

    PetscReal muRight = 0.0;
    flowParameters->muFunction.function(fieldR, auxR[aOff[T]], &muRight, flowParameters->muFunction.context.get());
    PetscReal kRight = 0.0;
    flowParameters->kFunction.function(fieldR, auxR[aOff[T]], &kRight, flowParameters->kFunction.context.get());

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
        flux[CompressibleFlowFields::RHOU + c] = viscousFlux;
    }

    // energy equation
    flux[CompressibleFlowFields::RHOE] = 0.0;
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

        flux[CompressibleFlowFields::RHOE] += heatFlux;
    }

    // zero out the density flux
    flux[CompressibleFlowFields::RHO] = 0.0;
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
                                                                                       const PetscScalar* conservedValues, const PetscInt aOff[], PetscScalar* auxField, void* ctx) {
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[uOff[0] + CompressibleFlowFields::RHO];

    for (PetscInt d = 0; d < dim; d++) {
        auxField[aOff[0] + d] = conservedValues[uOff[0] + CompressibleFlowFields::RHOU + d] / density;
    }

    PetscFunctionReturn(0);
}

// When used, you must request euler, then densityYi
PetscErrorCode ablate::finiteVolume::processes::EulerTransport::UpdateAuxTemperatureField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[],
                                                                                          const PetscScalar* conservedValues, const PetscInt aOff[], PetscScalar* auxField, void* ctx) {
    PetscFunctionBeginUser;
    auto flowParameters = (UpdateTemperatureData*)ctx;
    PetscErrorCode ierr = flowParameters->computeTemperatureFunction.function(conservedValues, *(auxField + aOff[0]), auxField + aOff[0], flowParameters->computeTemperatureFunction.context.get());
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

// When used, you must request euler, then densityYi
PetscErrorCode ablate::finiteVolume::processes::EulerTransport::UpdateAuxPressureField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[],
                                                                                       const PetscScalar* conservedValues, const PetscInt aOff[], PetscScalar* auxField, void* ctx) {
    PetscFunctionBeginUser;
    auto eulerAdvectionData = (AdvectionData*)ctx;

    // Get the speed of sound from the eos
    PetscReal temperature;
    eulerAdvectionData->computeTemperature.function(conservedValues, &temperature, eulerAdvectionData->computeTemperature.context.get()) >> checkError;
    eulerAdvectionData->computePressure.function(conservedValues, temperature, auxField + aOff[0], eulerAdvectionData->computePressure.context.get()) >> checkError;

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::EulerTransport, "build advection/diffusion for the euler field",
         OPT(ablate::parameters::Parameters, "parameters", "the parameters used by advection"), ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"),
         OPT(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculator", "the flux calculator (default is no advection)"),
         OPT(ablate::eos::transport::TransportModel, "transport", "the diffusion transport model (default is no diffusion)"));
