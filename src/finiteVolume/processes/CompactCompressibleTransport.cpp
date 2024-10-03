#include "CompactCompressibleTransport.hpp"
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
    }
    advectionData.numberSpecies = (PetscInt)eos->GetSpeciesVariables().size();
    timeStepData.advectionData = &advectionData;
    timeStepData.pgs = std::move(pgs);
    if(baseTransport) {
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
        flow.RegisterComputeTimeStepFunction(ablate::finiteVolume::processes::NavierStokesTransport::ComputeCflTimeStep, &timeStepData, "cfl");

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
            flow.RegisterRHSFunction(ablate::finiteVolume::processes::NavierStokesTransport::DiffusionFlux,
                                     &diffusionData,
                                     {CompressibleFlowFields::EULER_FIELD},
                                     {CompressibleFlowFields::EULER_FIELD},
                                     {CompressibleFlowFields::TEMPERATURE_FIELD, CompressibleFlowFields::VELOCITY_FIELD});
        }
        if (diffusionData.diffFunction.function) {
            // Specify a different rhs function depending on if the diffusion flux is constant
            if (diffusionData.diffFunction.propertySize == 1) {
                flow.RegisterRHSFunction(ablate::finiteVolume::processes::SpeciesTransport::DiffusionEnergyFlux,
                                         &diffusionData,
                                         {CompressibleFlowFields::EULER_FIELD},
                                         {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD},
                                         {CompressibleFlowFields::YI_FIELD, CompressibleFlowFields::TEMPERATURE_FIELD});
                flow.RegisterRHSFunction(ablate::finiteVolume::processes::SpeciesTransport::DiffusionSpeciesFlux,
                                         &diffusionData,
                                         {CompressibleFlowFields::DENSITY_YI_FIELD},
                                         {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD},
                                         {CompressibleFlowFields::YI_FIELD, CompressibleFlowFields::TEMPERATURE_FIELD});
            } else if (diffusionData.diffFunction.propertySize == advectionData.numberSpecies) {
                flow.RegisterRHSFunction(ablate::finiteVolume::processes::SpeciesTransport::DiffusionEnergyFluxVariableDiffusionCoefficient,
                                         &diffusionData,
                                         {CompressibleFlowFields::EULER_FIELD},
                                         {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD},
                                         {CompressibleFlowFields::YI_FIELD, CompressibleFlowFields::TEMPERATURE_FIELD});
                flow.RegisterRHSFunction(ablate::finiteVolume::processes::SpeciesTransport::DiffusionSpeciesFluxVariableDiffusionCoefficient,
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

            if (diffusionTimeStepData.conductionStabilityFactor > 0) {
                flow.RegisterComputeTimeStepFunction(ablate::finiteVolume::processes::NavierStokesTransport::ComputeConductionTimeStep, &diffusionTimeStepData, "cond");
            }
            if (diffusionTimeStepData.viscousStabilityFactor > 0) {
                flow.RegisterComputeTimeStepFunction(ablate::finiteVolume::processes::NavierStokesTransport::ComputeViscousDiffusionTimeStep, &diffusionTimeStepData, "visc");
            }
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
    if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::YI_FIELD)) {
        flow.RegisterAuxFieldUpdate(ablate::finiteVolume::processes::SpeciesTransport::UpdateAuxMassFractionField, &numberSpecies, {CompressibleFlowFields::YI_FIELD}, {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD});
        // clean up the species
        flow.RegisterPostEvaluate(ablate::finiteVolume::processes::SpeciesTransport::NormalizeSpecies);
    }
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
        //grab Temperature from aux field somehow and use that here, probably just auxL[0] and auxR[0] //I want to see how different tempL and it's calculated temperature are
        // If the same perfect, get rid of this step -klb
        PetscCall(advectionData->computeTemperature.function(fieldL, auxL[aOff[0]]*.66+.34*auxR[aOff[0]], &temperatureL, advectionData->computeTemperature.context.get()));
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

        PetscCall(advectionData->computeTemperature.function(fieldR, auxR[aOff[0]]*.66+.34*auxL[aOff[0]], &temperatureR, advectionData->computeTemperature.context.get()));

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
