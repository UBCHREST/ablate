#include "compactCompressibleTransport.hpp"
#include <utility>
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/fluxCalculator/ausm.hpp"
#include "finiteVolume/processes/navierStokesTransport.hpp"
#include "finiteVolume/processes/speciesTransport.hpp"
#include "parameters/emptyParameters.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscUtilities.hpp"

ablate::finiteVolume::processes::CompactCompressibleTransport::CompactCompressibleTransport(
        const std::shared_ptr<parameters::Parameters> &parametersIn, std::shared_ptr<eos::EOS> eosIn, std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalcIn,
        std::shared_ptr<eos::transport::TransportModel> baseTransport, std::shared_ptr<ablate::finiteVolume::processes::PressureGradientScaling> pgs)
        : advectionData(), fluxCalculator(fluxCalcIn), eos(std::move(eosIn)),transportModel(std::move(baseTransport)) {
    auto parameters = ablate::parameters::EmptyParameters::Check(parametersIn);

    if(fluxCalculator) {
        // cfl
        advectionData.cfl = parameters->Get<PetscReal>("cfl", 0.5);

        // extract the difference function from fluxDifferencer object
        advectionData.fluxCalculatorFunction = fluxCalculator->GetFluxCalculatorFunction();
        advectionData.fluxCalculatorCtx = fluxCalculator->GetFluxCalculatorContext();

        advectionData.numberSpecies = (PetscInt)eos->GetSpeciesVariables().size();
    }

    timeStepData.advectionData = &advectionData;
    timeStepData.pgs = std::move(pgs);

    if(transportModel) {
        // Add in the time stepping
        diffusionTimeStepData.conductionStabilityFactor = parameters->Get<PetscReal>("conductionStabilityFactor", 0.0);
        diffusionTimeStepData.viscousStabilityFactor = parameters->Get<PetscReal>("viscousStabilityFactor", 0.0);
        diffusionTimeStepData.diffusiveStabilityFactor = parameters->Get<PetscReal>("speciesStabilityFactor", 0.0);
        diffusionTimeStepData.speciesDiffusionCoefficient.resize(eos->GetSpeciesVariables().size());

        diffusionData.numberSpecies = (PetscInt)eos->GetSpeciesVariables().size();
        diffusionData.speciesSpeciesSensibleEnthalpy.resize(eos->GetSpeciesVariables().size());
        diffusionData.speciesDiffusionCoefficient.resize(eos->GetSpeciesVariables().size());
    }
}

void ablate::finiteVolume::processes::CompactCompressibleTransport::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    // Register the euler,eulerYi, and species source terms
    if (fluxCalculator) {
        //I don't know why we wouldn't push through the old temperature fields, maybe slower for perfect gas/idealized gas's but when there is a temperature iterative method this should be better
        //If it is worse for perfect gas's, going to need to add in an option switch -klb
        flow.RegisterRHSFunction(AdvectionFlux, &advectionData, {CompressibleFlowFields::EULER_FIELD,CompressibleFlowFields::DENSITY_YI_FIELD},
                                 {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD}, {CompressibleFlowFields::TEMPERATURE_FIELD});

        //Set the ComputeCFLTimestepFrom flow Process through
        flow.RegisterComputeTimeStepFunction(ComputeCflTimeStep, &timeStepData, "cfl");

        advectionData.computeTemperature = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Temperature, flow.GetSubDomain().GetFields());
        advectionData.computeInternalEnergy = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::InternalSensibleEnergy, flow.GetSubDomain().GetFields());
        advectionData.computeSpeedOfSound = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SpeedOfSound, flow.GetSubDomain().GetFields());
        advectionData.computePressure = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Pressure, flow.GetSubDomain().GetFields());
    }

    // if there are any coefficients for diffusion, compute diffusion, for here we will follow suit in allowing multiple diffusion functions

    if (transportModel) {
        // Store the required data for the low level c functions
        diffusionData.muFunction = transportModel->GetTransportTemperatureFunction(eos::transport::TransportProperty::Viscosity, flow.GetSubDomain().GetFields());
        diffusionData.kFunction = transportModel->GetTransportTemperatureFunction(eos::transport::TransportProperty::Conductivity, flow.GetSubDomain().GetFields());
        diffusionData.diffFunction = transportModel->GetTransportTemperatureFunction(eos::transport::TransportProperty::Diffusivity, flow.GetSubDomain().GetFields());
        diffusionData.computeSpeciesSensibleEnthalpyFunction = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SpeciesSensibleEnthalpy, flow.GetSubDomain().GetFields());

        /* For now we will do 3 different diffusive flux calls just since it doesn't seem like their splitting costs too much time in redundant calculations */
        if (diffusionData.muFunction.function || diffusionData.kFunction.function) {
            // Register the Diffusion Source term
            flow.RegisterRHSFunction(DiffusionFlux,
                                     &diffusionData,
                                     {CompressibleFlowFields::EULER_FIELD},
                                     {CompressibleFlowFields::EULER_FIELD},
                                     {CompressibleFlowFields::TEMPERATURE_FIELD, CompressibleFlowFields::VELOCITY_FIELD});
        }
        if (diffusionData.diffFunction.function) {
            // Specify a different rhs function depending on if the diffusion flux is constant
            if (diffusionData.diffFunction.propertySize == 1) {
                flow.RegisterRHSFunction(DiffusionEnergyFlux,
                                         &diffusionData,
                                         {CompressibleFlowFields::EULER_FIELD},
                                         {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD},
                                         {CompressibleFlowFields::YI_FIELD, CompressibleFlowFields::TEMPERATURE_FIELD});
                flow.RegisterRHSFunction(DiffusionSpeciesFlux,
                                         &diffusionData,
                                         {CompressibleFlowFields::DENSITY_YI_FIELD},
                                         {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD},
                                         {CompressibleFlowFields::YI_FIELD, CompressibleFlowFields::TEMPERATURE_FIELD});
            } else if (diffusionData.diffFunction.propertySize == advectionData.numberSpecies) {
                flow.RegisterRHSFunction(DiffusionEnergyFluxVariableDiffusionCoefficient,
                                         &diffusionData,
                                         {CompressibleFlowFields::EULER_FIELD},
                                         {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD},
                                         {CompressibleFlowFields::YI_FIELD, CompressibleFlowFields::TEMPERATURE_FIELD});
                flow.RegisterRHSFunction(DiffusionSpeciesFluxVariableDiffusionCoefficient,
                                         &diffusionData,
                                         {CompressibleFlowFields::DENSITY_YI_FIELD},
                                         {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD},
                                         {CompressibleFlowFields::YI_FIELD, CompressibleFlowFields::TEMPERATURE_FIELD});
            } else {
                throw std::invalid_argument("The diffusion property size must be 1 or number of species in ablate::finiteVolume::processes::SpeciesTransport.");
            }
        }

        // Check to see if time step calculations should be added for viscosity or conduction
        if (diffusionTimeStepData.conductionStabilityFactor > 0 || diffusionTimeStepData.viscousStabilityFactor > 0) {
            diffusionTimeStepData.kFunction = diffusionData.kFunction;
            diffusionTimeStepData.muFunction = diffusionData.muFunction;
            diffusionTimeStepData.specificHeat = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SpecificHeatConstantVolume, flow.GetSubDomain().GetFields());
            diffusionTimeStepData.density = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Density, flow.GetSubDomain().GetFields());

            diffusionTimeStepData.numberSpecies = eos->GetSpeciesVariables().size();

            if (diffusionTimeStepData.conductionStabilityFactor > 0) {
                flow.RegisterComputeTimeStepFunction(ComputeConductionTimeStep, &diffusionTimeStepData, "cond");
            }
            if (diffusionTimeStepData.viscousStabilityFactor > 0) {
                flow.RegisterComputeTimeStepFunction(ComputeViscousDiffusionTimeStep, &diffusionTimeStepData, "visc");
            }
        }
        if (diffusionTimeStepData.diffusiveStabilityFactor > 0){
                    diffusionTimeStepData.numberSpecies = diffusionData.numberSpecies;
                    diffusionTimeStepData.diffFunction = diffusionData.diffFunction;
                    flow.RegisterComputeTimeStepFunction(ComputeViscousSpeciesDiffusionTimeStep, &diffusionTimeStepData, "spec");
        }
    }

    //Setup up aux updates and normalizations
    if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::VELOCITY_FIELD)) {
        flow.RegisterAuxFieldUpdate(ablate::finiteVolume::processes::NavierStokesTransport::UpdateAuxVelocityField, nullptr, std::vector<std::string>{CompressibleFlowFields::VELOCITY_FIELD}, {CompressibleFlowFields::EULER_FIELD});
    }
    if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::TEMPERATURE_FIELD)) {
        computeTemperatureFunction = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Temperature, flow.GetSubDomain().GetFields());
        flow.RegisterAuxFieldUpdate(ablate::finiteVolume::processes::NavierStokesTransport::UpdateAuxTemperatureField, &computeTemperatureFunction, std::vector<std::string>{CompressibleFlowFields::TEMPERATURE_FIELD}, {});
    }
    if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::PRESSURE_FIELD)) {
        computePressureFunction = eos->GetThermodynamicFunction(eos::ThermodynamicProperty::Pressure, flow.GetSubDomain().GetFields());
        flow.RegisterAuxFieldUpdate(ablate::finiteVolume::processes::NavierStokesTransport::UpdateAuxPressureField, &computePressureFunction, std::vector<std::string>{CompressibleFlowFields::PRESSURE_FIELD}, {});
    }
    flow.RegisterAuxFieldUpdate(ablate::finiteVolume::processes::SpeciesTransport::UpdateAuxMassFractionField, &advectionData.numberSpecies, std::vector<std::string>{CompressibleFlowFields::YI_FIELD}, {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD});
    // clean up the species
    flow.RegisterPostEvaluate(ablate::finiteVolume::processes::SpeciesTransport::NormalizeSpecies);
}

PetscErrorCode ablate::finiteVolume::processes::CompactCompressibleTransport::AdvectionFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt* uOff, const PetscScalar* fieldL,
                                                                                     const PetscScalar* fieldR, const PetscInt* aOff, const PetscScalar* auxL, const PetscScalar* auxR,
                                                                                     PetscScalar* flux, void* ctx) {
    PetscFunctionBeginUser;

    auto advectionData = (AdvectionData*)ctx;

    const int EULER_FIELD = 0;
    const int RHOYI_FIELD = 1;

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

        PetscCall(advectionData->computeTemperature.function(fieldL, auxL[aOff[0]]*.67+.33*auxR[aOff[0]], &temperatureL, advectionData->computeTemperature.context.get()));

        // Get the velocity in this direction
        normalVelocityL = 0.0;
        for (PetscInt d = 0; d < dim; d++) {
            velocityL[d] = fieldL[uOff[EULER_FIELD] + CompressibleFlowFields::RHOU + d] / densityL;
            normalVelocityL += velocityL[d] * norm[d];
        }
        PetscCall(advectionData->computeInternalEnergy.function(fieldL, temperatureL, &internalEnergyL, advectionData->computeInternalEnergy.context.get()));
        PetscCall(advectionData->computeSpeedOfSound.function(fieldL, temperatureL, &aL, advectionData->computeSpeedOfSound.context.get()));
        PetscCall(advectionData->computePressure.function(fieldL, temperatureL, &pL, advectionData->computePressure.context.get()));
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

        PetscCall(advectionData->computeTemperature.function(fieldR, auxR[aOff[0]]*.67+.33*auxL[aOff[0]], &temperatureR, advectionData->computeTemperature.context.get()));

        // Get the velocity in this direction
        normalVelocityR = 0.0;
        for (PetscInt d = 0; d < dim; d++) {
            velocityR[d] = fieldR[uOff[EULER_FIELD] + CompressibleFlowFields::RHOU + d] / densityR;
            normalVelocityR += velocityR[d] * norm[d];
        }

        PetscCall(advectionData->computeInternalEnergy.function(fieldR, temperatureR, &internalEnergyR, advectionData->computeInternalEnergy.context.get()));
        PetscCall(advectionData->computeSpeedOfSound.function(fieldR, temperatureR, &aR, advectionData->computeSpeedOfSound.context.get()));
        PetscCall(advectionData->computePressure.function(fieldR, temperatureR, &pR, advectionData->computePressure.context.get()));
    }

    // get the face values
    PetscReal massFlux;
    PetscReal p12;

    fluxCalculator::Direction direction =
        advectionData->fluxCalculatorFunction(advectionData->fluxCalculatorCtx, normalVelocityL, aL, densityL, pL, normalVelocityR, aR, densityR, pR, &massFlux, &p12);

    if (direction == fluxCalculator::LEFT) {
        flux[CompressibleFlowFields::RHO] = massFlux * areaMag;
        PetscReal velMagL = utilities::MathUtilities::MagVector(dim, velocityL);
        PetscReal HL = internalEnergyL + velMagL * velMagL / 2.0 + pL / densityL;
        flux[CompressibleFlowFields::RHOE] = HL * massFlux * areaMag;
        for (PetscInt n = 0; n < dim; n++) {
            flux[CompressibleFlowFields::RHOU + n] = velocityL[n] * massFlux * areaMag + p12 * fg->normal[n];
        }
        for (PetscInt ns = 0; ns < advectionData->numberSpecies; ns++)
            flux[uOff[RHOYI_FIELD]+ns] = massFlux * fieldL[uOff[RHOYI_FIELD] + ns] / densityL * areaMag;
    } else if (direction == fluxCalculator::RIGHT) {
        flux[CompressibleFlowFields::RHO] = massFlux * areaMag;
        PetscReal velMagR = utilities::MathUtilities::MagVector(dim, velocityR);
        PetscReal HR = internalEnergyR + velMagR * velMagR / 2.0 + pR / densityR;
        flux[CompressibleFlowFields::RHOE] = HR * massFlux * areaMag;
        for (PetscInt n = 0; n < dim; n++) {
            flux[CompressibleFlowFields::RHOU + n] = velocityR[n] * massFlux * areaMag + p12 * fg->normal[n];
        }
        for (PetscInt ns = 0; ns < advectionData->numberSpecies; ns++)
            flux[uOff[RHOYI_FIELD]+ns] = massFlux * fieldR[uOff[RHOYI_FIELD] + ns] / densityR * areaMag;
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
        for (PetscInt ns = 0; ns < advectionData->numberSpecies; ns++)
            flux[uOff[RHOYI_FIELD]+ns] = massFlux * 0.5 * (fieldR[uOff[RHOYI_FIELD] + ns] + fieldL[uOff[RHOYI_FIELD] + ns] ) / ( 0.5 * (densityL + densityR)) * areaMag;
    }

    PetscFunctionReturn(0);
}


double ablate::finiteVolume::processes::CompactCompressibleTransport::ComputeCflTimeStep(TS ts, ablate::finiteVolume::FiniteVolumeSolver& flow, void* ctx) {
    // Get the dm and current solution vector
    DM dm;
    TSGetDM(ts, &dm) >> utilities::PetscUtilities::checkError;
    Vec v;
    TSGetSolution(ts, &v) >> utilities::PetscUtilities::checkError;

    // Get the flow param
    auto timeStepData = (CflTimeStepData*)ctx;
    auto advectionData = timeStepData->advectionData;

    // Get the fv geom
    Vec locCharacteristicsVec;
    DM characteristicsDm;
    const PetscScalar* locCharacteristicsArray;
    flow.GetMeshCharacteristics(characteristicsDm, locCharacteristicsVec);
    VecGetArrayRead(locCharacteristicsVec, &locCharacteristicsArray) >> utilities::PetscUtilities::checkError;

    // Get the valid cell range over this region
    ablate::domain::Range cellRange;
    flow.GetCellRangeWithoutGhost(cellRange);

    const PetscScalar* x;
    VecGetArrayRead(v, &x) >> utilities::PetscUtilities::checkError;

    // Get the dim from the dm
    PetscInt dim;
    DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;

    // Get field location for euler and densityYi
    auto eulerId = flow.GetSubDomain().GetField("euler").id;

    // Get alpha if provided
    PetscReal pgsAlpha = 1.0;
    if (timeStepData->pgs) {
        pgsAlpha = timeStepData->pgs->GetAlpha();
    }

    // March over each cell
    PetscReal dtMin = ablate::utilities::Constants::large;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        auto cell = cellRange.GetPoint(c);

        const PetscReal* euler;
        const PetscReal* conserved = NULL;
        const PetscReal* cellCharacteristics = NULL;
        DMPlexPointGlobalFieldRead(dm, cell, eulerId, x, &euler) >> utilities::PetscUtilities::checkError;
        DMPlexPointGlobalRead(dm, cell, x, &conserved) >> utilities::PetscUtilities::checkError;
        DMPlexPointLocalRead(characteristicsDm, cell, locCharacteristicsArray, &cellCharacteristics) >> utilities::PetscUtilities::checkError;

        if (euler) {  // must be real cell and not ghost
            PetscReal rho = euler[CompressibleFlowFields::RHO];

            // Get the speed of sound from the eos
            //TODO:: Replace this with a better temperature guess (see compute conduction Time Step below)
            PetscReal temperature;
            advectionData->computeTemperature.function(conserved, 300, &temperature, advectionData->computeTemperature.context.get()) >> utilities::PetscUtilities::checkError;
            PetscReal a;
            advectionData->computeSpeedOfSound.function(conserved, temperature, &a, advectionData->computeSpeedOfSound.context.get()) >> utilities::PetscUtilities::checkError;

            PetscReal dx = 2.0 * cellCharacteristics[FiniteVolumeSolver::MIN_CELL_RADIUS];

            PetscReal velSum = 0.0;
            for (PetscInt d = 0; d < dim; d++) {
                velSum += PetscAbsReal(euler[CompressibleFlowFields::RHOU + d]) / rho;
            }
            PetscReal dt = advectionData->cfl * dx / (a / pgsAlpha + velSum);

            dtMin = PetscMin(dtMin, dt);
        }
    }
    VecRestoreArrayRead(v, &x) >> utilities::PetscUtilities::checkError;
    flow.RestoreRange(cellRange);
    VecRestoreArrayRead(locCharacteristicsVec, &locCharacteristicsArray) >> utilities::PetscUtilities::checkError;

    return dtMin;
}

double ablate::finiteVolume::processes::CompactCompressibleTransport::ComputeConductionTimeStep(TS ts, ablate::finiteVolume::FiniteVolumeSolver& flow, void* ctx) {
    // Get the dm and current solution vector
    DM dm;
    TSGetDM(ts, &dm) >> utilities::PetscUtilities::checkError;
    Vec v;
    TSGetSolution(ts, &v) >> utilities::PetscUtilities::checkError;

    // Get the flow param
    auto diffusionData = (DiffusionTimeStepData*)ctx;

    // Get the fv geom
    Vec locCharacteristicsVec;
    DM characteristicsDm;
    const PetscScalar* locCharacteristicsArray;
    flow.GetMeshCharacteristics(characteristicsDm, locCharacteristicsVec);
    VecGetArrayRead(locCharacteristicsVec, &locCharacteristicsArray) >> utilities::PetscUtilities::checkError;

    // Get the valid cell range over this region
    ablate::domain::Range cellRange;
    flow.GetCellRangeWithoutGhost(cellRange);

    // Get the solution data
    const PetscScalar* x;
    VecGetArrayRead(v, &x) >> utilities::PetscUtilities::checkError;

    // Get the auxData
    const PetscScalar* aux;
    const DM auxDM = flow.GetSubDomain().GetAuxDM();
    VecGetArrayRead(flow.GetSubDomain().GetAuxGlobalVector(), &aux) >> utilities::PetscUtilities::checkError;

    // Get the dim from the dm
    PetscInt dim;
    DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;

    // Get field location for temperature
    auto temperatureField = flow.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD).id;

    // get functions
    auto kFunction = diffusionData->kFunction.function;
    auto kFunctionContext = diffusionData->kFunction.context.get();
    auto cvFunction = diffusionData->specificHeat.function;
    auto cvFunctionContext = diffusionData->specificHeat.context.get();
    auto density = diffusionData->density.function;
    auto densityContext = diffusionData->density.context.get();
    auto stabFactor = diffusionData->conductionStabilityFactor;

    // March over each cell
    PetscReal dtMin = ablate::utilities::Constants::large;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        auto cell = cellRange.GetPoint(c);

        const PetscReal* conserved = NULL;
        DMPlexPointGlobalRead(dm, cell, x, &conserved) >> utilities::PetscUtilities::checkError;

        const PetscReal* temperature = NULL;
        DMPlexPointLocalFieldRead(auxDM, cell, temperatureField, aux, &temperature) >> utilities::PetscUtilities::checkError;

        const PetscReal* cellCharacteristics = NULL;
        DMPlexPointLocalRead(characteristicsDm, cell, locCharacteristicsArray, &cellCharacteristics) >> utilities::PetscUtilities::checkError;

        if (conserved) {  // must be real cell and not ghost
            PetscReal k;
            kFunction(conserved, *temperature, &k, kFunctionContext) >> utilities::PetscUtilities::checkError;
            PetscReal cv;
            cvFunction(conserved, *temperature, &cv, cvFunctionContext) >> utilities::PetscUtilities::checkError;
            PetscReal rho;
            density(conserved, *temperature, &rho, densityContext) >> utilities::PetscUtilities::checkError;

            // Compute alpha
            PetscReal alpha = k / (rho * cv);

            PetscReal dx2 = PetscSqr(2.0 * cellCharacteristics[FiniteVolumeSolver::MIN_CELL_RADIUS]);

            // compute dt
            double dt = PetscAbs(stabFactor * dx2 / alpha);
            dtMin = PetscMin(dtMin, dt);
        }
    }
    VecRestoreArrayRead(v, &x) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(flow.GetSubDomain().GetAuxGlobalVector(), &aux) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(locCharacteristicsVec, &locCharacteristicsArray) >> utilities::PetscUtilities::checkError;
    flow.RestoreRange(cellRange);

    return dtMin;
}

double ablate::finiteVolume::processes::CompactCompressibleTransport::ComputeViscousDiffusionTimeStep(TS ts, ablate::finiteVolume::FiniteVolumeSolver& flow, void* ctx) {
    // Get the dm and current solution vector
    DM dm;
    TSGetDM(ts, &dm) >> utilities::PetscUtilities::checkError;
    Vec v;
    TSGetSolution(ts, &v) >> utilities::PetscUtilities::checkError;

    // Get the flow param
    auto diffusionData = (DiffusionTimeStepData*)ctx;

    // Get the fv geom
    Vec locCharacteristicsVec;
    DM characteristicsDm;
    const PetscScalar* locCharacteristicsArray;
    flow.GetMeshCharacteristics(characteristicsDm, locCharacteristicsVec);
    VecGetArrayRead(locCharacteristicsVec, &locCharacteristicsArray) >> utilities::PetscUtilities::checkError;

    // Get the valid cell range over this region
    ablate::domain::Range cellRange;
    flow.GetCellRangeWithoutGhost(cellRange);

    // Get the solution data
    const PetscScalar* x;
    VecGetArrayRead(v, &x) >> utilities::PetscUtilities::checkError;

    // Get the auxData
    const PetscScalar* aux;
    const DM auxDM = flow.GetSubDomain().GetAuxDM();
    VecGetArrayRead(flow.GetSubDomain().GetAuxGlobalVector(), &aux) >> utilities::PetscUtilities::checkError;

    // Get the dim from the dm
    PetscInt dim;
    DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;

    // Get field location for temperature
    auto temperatureField = flow.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD).id;

    // get functions
    auto muFunction = diffusionData->muFunction.function;
    auto muFunctionContext = diffusionData->muFunction.context.get();
    auto density = diffusionData->density.function;
    auto densityContext = diffusionData->density.context.get();
    auto stabFactor = diffusionData->viscousStabilityFactor;

    // March over each cell
    PetscReal dtMin = ablate::utilities::Constants::large;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        auto cell = cellRange.GetPoint(c);

        const PetscReal* conserved = NULL;
        DMPlexPointGlobalRead(dm, cell, x, &conserved) >> utilities::PetscUtilities::checkError;

        const PetscReal* temperature = NULL;
        DMPlexPointLocalFieldRead(auxDM, cell, temperatureField, aux, &temperature) >> utilities::PetscUtilities::checkError;

        const PetscReal* cellCharacteristics = NULL;
        DMPlexPointLocalRead(characteristicsDm, cell, locCharacteristicsArray, &cellCharacteristics) >> utilities::PetscUtilities::checkError;

        if (conserved) {  // must be real cell and not ghost
            PetscReal mu;
            muFunction(conserved, *temperature, &mu, muFunctionContext) >> utilities::PetscUtilities::checkError;
            PetscReal rho;
            density(conserved, *temperature, &rho, densityContext) >> utilities::PetscUtilities::checkError;

            // Compute nu
            PetscReal nu = mu / rho;

            PetscReal dx2 = PetscSqr(2.0 * cellCharacteristics[FiniteVolumeSolver::MIN_CELL_RADIUS]);

            // compute dt
            double dt = PetscAbs(stabFactor * dx2 / nu);
            dtMin = PetscMin(dtMin, dt);
        }
    }
    VecRestoreArrayRead(v, &x) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(flow.GetSubDomain().GetAuxGlobalVector(), &aux) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(locCharacteristicsVec, &locCharacteristicsArray) >> utilities::PetscUtilities::checkError;
    flow.RestoreRange(cellRange);
    return dtMin;
}
double ablate::finiteVolume::processes::CompactCompressibleTransport::ComputeViscousSpeciesDiffusionTimeStep(TS ts, ablate::finiteVolume::FiniteVolumeSolver &flow, void *ctx) {
    // Get the dm and current solution vector
    DM dm;
    TSGetDM(ts, &dm) >> utilities::PetscUtilities::checkError;
    Vec v;
    TSGetSolution(ts, &v) >> utilities::PetscUtilities::checkError;

    // Get the flow param
    auto diffusionData = (DiffusionTimeStepData *)ctx;

    // Get the fv geom
    PetscReal minCellRadius;
    DMPlexGetGeometryFVM(dm, NULL, NULL, &minCellRadius) >> utilities::PetscUtilities::checkError;

    // Get the valid cell range over this region
    ablate::domain::Range cellRange;
    flow.GetCellRangeWithoutGhost(cellRange);

    // Get the solution data
    const PetscScalar *x;
    VecGetArrayRead(v, &x) >> utilities::PetscUtilities::checkError;

    // Get the auxData
    const PetscScalar *aux;
    const DM auxDM = flow.GetSubDomain().GetAuxDM();
    VecGetArrayRead(flow.GetSubDomain().GetAuxGlobalVector(), &aux) >> utilities::PetscUtilities::checkError;

    // Get the dim from the dm
    PetscInt dim;
    DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;

    // assume the smallest cell is the limiting factor for now
    const PetscReal dx2 = PetscSqr(2.0 * minCellRadius);

    // Get field location for temperature
    auto temperatureField = flow.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD).id;

    // get functions
    auto diffFunction = diffusionData->diffFunction.function;
    auto diffFunctionContext = diffusionData->diffFunction.context.get();
    auto stabFactor = diffusionData->diffusiveStabilityFactor;

    // March over each cell
    PetscReal dtMin = ablate::utilities::Constants::large;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        auto cell = cellRange.GetPoint(c);

        const PetscReal *conserved = NULL;
        DMPlexPointGlobalRead(dm, cell, x, &conserved) >> utilities::PetscUtilities::checkError;

        const PetscReal *temperature = NULL;
        DMPlexPointLocalFieldRead(auxDM, cell, temperatureField, aux, &temperature) >> utilities::PetscUtilities::checkError;

        if (conserved) {  // must be real cell and not ghost
            PetscReal diff;
            diffFunction(conserved, *temperature, &diff, diffFunctionContext) >> utilities::PetscUtilities::checkError;

            // compute dt
            double dt = PetscAbs(stabFactor * dx2 / diff);
            dtMin = PetscMin(dtMin, dt);
        }
    }
    VecRestoreArrayRead(v, &x) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(flow.GetSubDomain().GetAuxGlobalVector(), &aux) >> utilities::PetscUtilities::checkError;
    flow.RestoreRange(cellRange);
    return dtMin;
}

PetscErrorCode ablate::finiteVolume::processes::CompactCompressibleTransport::DiffusionFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[],
                                                                                     const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[],
                                                                                     const PetscScalar gradAux[], PetscScalar flux[], void* ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int T = 0;
    const int VEL = 1;

    auto flowParameters = (DiffusionData*)ctx;

    // Compute mu and k
    PetscReal mu = 0.0;
    flowParameters->muFunction.function(field, aux[aOff[T]], &mu, flowParameters->muFunction.context.get());
    PetscReal k = 0.0;
    flowParameters->kFunction.function(field, aux[aOff[T]], &k, flowParameters->kFunction.context.get());

    // Compute the stress tensor tau
    PetscReal tau[9];  // Maximum size without symmetry
    PetscCall(ablate::finiteVolume::processes::NavierStokesTransport::CompressibleFlowComputeStressTensor(dim, mu, gradAux + aOff_x[VEL], tau));

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
            heatFlux += aux[aOff[VEL] + c] * tau[d * dim + c];
        }

        // heat conduction (-k dT/dx - k dT/dy - k dT/dz) . n A
        heatFlux += k * gradAux[aOff_x[T] + d];

        // Multiply by the area normal
        heatFlux *= -fg->normal[d];

        flux[CompressibleFlowFields::RHOE] += heatFlux;
    }

    // zero out the density flux
    flux[CompressibleFlowFields::RHO] = 0.0;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::CompactCompressibleTransport::DiffusionEnergyFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscInt uOff_x[],
                                                                                      const PetscScalar field[], const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[],
                                                                                      const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void *ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int yi = 0;
    const int euler = 0;
    const int temp = 1;

    auto flowParameters = (DiffusionData *)ctx;

    // get the current density from euler
    const PetscReal density = field[uOff[euler] + CompressibleFlowFields::RHO];

    // compute the temperature in this volume
    const PetscReal temperature = aux[aOff[temp]];
    PetscCall(flowParameters->computeSpeciesSensibleEnthalpyFunction.function(
        field, temperature, flowParameters->speciesSpeciesSensibleEnthalpy.data(), flowParameters->computeSpeciesSensibleEnthalpyFunction.context.get()));

    // set the non rho E fluxes to zero
    flux[CompressibleFlowFields::RHO] = 0.0;
    flux[CompressibleFlowFields::RHOE] = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        flux[CompressibleFlowFields::RHOU + d] = 0.0;
    }

    // compute diff, this can be constant or variable
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

PetscErrorCode ablate::finiteVolume::processes::CompactCompressibleTransport::DiffusionEnergyFluxVariableDiffusionCoefficient(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[],
                                                                                                                  const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
                                                                                                                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[],
                                                                                                                  const PetscScalar gradAux[], PetscScalar flux[], void *ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int yi = 0;
    const int euler = 0;
    const int temp = 1;

    auto flowParameters = (DiffusionData *)ctx;

    // get the current density from euler
    const PetscReal density = field[uOff[euler] + CompressibleFlowFields::RHO];

    // compute the temperature in this volume
    const PetscReal temperature = aux[aOff[temp]];
    PetscCall(flowParameters->computeSpeciesSensibleEnthalpyFunction.function(
        field, temperature, flowParameters->speciesSpeciesSensibleEnthalpy.data(), flowParameters->computeSpeciesSensibleEnthalpyFunction.context.get()));

    // set the non rho E fluxes to zero
    flux[CompressibleFlowFields::RHO] = 0.0;
    flux[CompressibleFlowFields::RHOE] = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        flux[CompressibleFlowFields::RHOU + d] = 0.0;
    }

    // compute diff, this can be constant or variable
    flowParameters->diffFunction.function(field, temperature, flowParameters->speciesDiffusionCoefficient.data(), flowParameters->diffFunction.context.get());

    for (PetscInt sp = 0; sp < flowParameters->numberSpecies; ++sp) {
        for (PetscInt d = 0; d < dim; ++d) {
            // speciesFlux(-rho Di dYi/dx - rho Di dYi/dy - rho Di dYi//dz) . n A
            const int offset = aOff_x[yi] + (sp * dim) + d;
            PetscReal speciesFlux = -fg->normal[d] * density * flowParameters->speciesDiffusionCoefficient[sp] * flowParameters->speciesSpeciesSensibleEnthalpy[sp] * gradAux[offset];
            flux[CompressibleFlowFields::RHOE] += speciesFlux;
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::CompactCompressibleTransport::DiffusionSpeciesFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscInt uOff_x[],
                                                                                       const PetscScalar field[], const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[],
                                                                                       const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void *ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int yi = 0;
    const int euler = 0;
    const int temp = 1;

    auto flowParameters = (DiffusionData *)ctx;

    // get the current density from euler
    const PetscReal density = field[uOff[euler] + CompressibleFlowFields::RHO];

    const PetscReal temperature = aux[aOff[temp]];

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

PetscErrorCode ablate::finiteVolume::processes::CompactCompressibleTransport::DiffusionSpeciesFluxVariableDiffusionCoefficient(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[],
                                                                                                                   const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
                                                                                                                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[],
                                                                                                                   const PetscScalar gradAux[], PetscScalar flux[], void *ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int yi = 0;
    const int euler = 0;
    const int temp = 1;

    auto flowParameters = (DiffusionData *)ctx;

    // get the current density from euler
    const PetscReal density = field[uOff[euler] + CompressibleFlowFields::RHO];
    const PetscReal temperature = aux[aOff[temp]];

    // compute diff
    flowParameters->diffFunction.function(field, temperature, flowParameters->speciesDiffusionCoefficient.data(), flowParameters->diffFunction.context.get());

    // species equations
    for (PetscInt sp = 0; sp < flowParameters->numberSpecies; ++sp) {
        flux[sp] = 0;
        for (PetscInt d = 0; d < dim; ++d) {
            // speciesFlux(-rho Di dYi/dx - rho Di dYi/dy - rho Di dYi//dz) . n A
            const int offset = aOff_x[yi] + (sp * dim) + d;
            PetscReal speciesFlux = -fg->normal[d] * density * flowParameters->speciesDiffusionCoefficient[sp] * gradAux[offset];
            flux[sp] += speciesFlux;
        }
    }

    PetscFunctionReturn(0);
}

