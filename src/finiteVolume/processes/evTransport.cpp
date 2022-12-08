#include "evTransport.hpp"
#include <utilities/mathUtilities.hpp>
#include "finiteVolume/compressibleFlowFields.hpp"

ablate::finiteVolume::processes::EVTransport::EVTransport(std::shared_ptr<eos::EOS> eosIn, std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalcIn,
                                                          std::shared_ptr<eos::transport::TransportModel> transportModelIn)
    : fluxCalculator(std::move(fluxCalcIn)), eos(std::move(eosIn)), transportModel(std::move(transportModelIn)) {}

void ablate::finiteVolume::processes::EVTransport::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    const auto &evConservedFields = flow.GetSubDomain().GetFields(domain::FieldLocation::SOL, CompressibleFlowFields::EV_TAG);

    for (auto &evConservedField : evConservedFields) {
        // increase the size of the stored data
        auto &advectionData = advectionDatas.emplace_back();
        auto &numberEV = numberEVs.emplace_back();
        auto &diffusionData = diffusionDatas.emplace_back();

        // determine the number of components in the ev
        advectionData.numberEV = evConservedField.numberComponents;
        numberEV = evConservedField.numberComponents;
        diffusionData.numberEV = evConservedField.numberComponents;

        // Get the nonConserved form
        auto nonConserved = evConservedField.name.substr(CompressibleFlowFields::CONSERVED.length());

        if (!flow.GetSubDomain().ContainsField(nonConserved)) {
            throw std::invalid_argument("The ablate::finiteVolume::processes::EVTransport process expects the conserved (" + evConservedField.name + ") and non-conserved (" + nonConserved +
                                        ") extra variables to be in the flow.");
        }

        if (fluxCalculator) {
            // extract the difference function from fluxDifferencer object
            advectionData.fluxCalculatorFunction = fluxCalculator->GetFluxCalculatorFunction();
            advectionData.fluxCalculatorCtx = fluxCalculator->GetFluxCalculatorContext();

            // set decode state functions
            advectionData.computeTemperature = eos->GetThermodynamicFunction(eos::ThermodynamicProperty::Temperature, flow.GetSubDomain().GetFields());
            advectionData.computeInternalEnergy = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::InternalSensibleEnergy, flow.GetSubDomain().GetFields());
            advectionData.computeSpeedOfSound = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SpeedOfSound, flow.GetSubDomain().GetFields());
            advectionData.computePressure = eos->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Pressure, flow.GetSubDomain().GetFields());

            flow.RegisterRHSFunction(AdvectionFlux, &advectionData, evConservedField.name, {CompressibleFlowFields::EULER_FIELD, evConservedField.name}, {});
        }

        if (transportModel) {
            diffusionData.speciesSpeciesSensibleEnthalpy.resize(eos->GetSpecies().size());

            diffusionData.diffFunction = transportModel->GetTransportFunction(eos::transport::TransportProperty::Diffusivity, flow.GetSubDomain().GetFields());

            if (diffusionData.diffFunction.function) {
                if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::YI_FIELD)) {
                    flow.RegisterRHSFunction(DiffusionEVFlux,
                                             &diffusionData,
                                             evConservedField.name,
                                             {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_YI_FIELD},
                                             {nonConserved, CompressibleFlowFields::YI_FIELD});
                } else {
                    flow.RegisterRHSFunction(DiffusionEVFlux, &diffusionData, evConservedField.name, {CompressibleFlowFields::EULER_FIELD}, {nonConserved});
                }
            }
        } else {
            diffusionData.diffFunction.function = nullptr;
        }

        flow.RegisterAuxFieldUpdate(UpdateEVField, &numberEV, std::vector<std::string>{nonConserved}, {CompressibleFlowFields::EULER_FIELD, evConservedField.name});
    }
}

PetscErrorCode ablate::finiteVolume::processes::EVTransport::UpdateEVField(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt *uOff, const PetscScalar *conservedValues,
                                                                           const PetscInt *aOff, PetscScalar *auxField, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[uOff[0] + CompressibleFlowFields::RHO];

    auto numberEV = (PetscInt *)ctx;

    for (PetscInt e = 0; e < *numberEV; e++) {
        auxField[aOff[0] + e] = conservedValues[uOff[1] + e] / density;
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::EVTransport::AdvectionFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscScalar *fieldL, const PetscScalar *fieldR,
                                                                           const PetscInt *aOff, const PetscScalar *auxL, const PetscScalar *auxR, PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
    auto eulerAdvectionData = (AdvectionData *)ctx;

    // Compute the norm
    PetscReal norm[3];
    utilities::MathUtilities::NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = utilities::MathUtilities::MagVector(dim, fg->normal);

    const int EULER_FIELD = 0;
    const int DENSITY_EV_FIELD = 1;

    // Decode the left and right states
    PetscReal densityL;
    PetscReal normalVelocityL;
    PetscReal velocityL[3];
    PetscReal internalEnergyL;
    PetscReal aL;
    PetscReal pL;
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

PetscErrorCode ablate::finiteVolume::processes::EVTransport::DiffusionEVFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[],
                                                                             const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[],
                                                                             const PetscScalar gradAux[], PetscScalar flux[], void *ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int EULER_FIELD = 0;
    const int EV_FIELD = 0;

    auto flowParameters = (DiffusionData *)ctx;

    // get the current density from euler
    const PetscReal density = field[uOff[EULER_FIELD] + CompressibleFlowFields::RHO];

    // compute diff
    PetscReal diff = 0.0;
    flowParameters->diffFunction.function(field, &diff, flowParameters->diffFunction.context.get());

    // species equations
    for (PetscInt ev = 0; ev < flowParameters->numberEV; ++ev) {
        flux[ev] = 0;
        for (PetscInt d = 0; d < dim; ++d) {
            // speciesFlux(-rho Di dYi/dx - rho Di dYi/dy - rho Di dYi//dz) . n A
            const int offset = aOff_x[EV_FIELD] + (ev * dim) + d;
            PetscReal evFlux = -fg->normal[d] * density * diff * gradAux[offset];
            flux[ev] += evFlux;
        }
    }

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::EVTransport, "diffusion/advection for the specified EV",
         ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"),
         OPT(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculator", "the flux calculator (default is no advection)"),
         OPT(ablate::eos::transport::TransportModel, "transport", "the diffusion transport model (default is no diffusion)"));
