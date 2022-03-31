#include "stiffenedGas.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"

ablate::eos::StiffenedGas::StiffenedGas(std::shared_ptr<ablate::parameters::Parameters> parametersIn, std::vector<std::string> species) : EOS("stiffenedGas"), species(species) {
    // set default values for options
    parameters.gamma = parametersIn->Get<PetscReal>("gamma", 1.932);
    parameters.Cp = parametersIn->Get<PetscReal>("Cp", 8095.08);
    parameters.p0 = parametersIn->Get<PetscReal>("p0", 1.1645e9);
    parameters.numberSpecies = species.size();
}

void ablate::eos::StiffenedGas::View(std::ostream &stream) const {
    stream << "EOS: " << type << std::endl;
    stream << "\tgamma: " << parameters.gamma << std::endl;
    stream << "\tCp: " << parameters.Cp << std::endl;
    stream << "\tp0: " << parameters.p0 << std::endl;
    if (!species.empty()) {
        stream << "\tspecies: " << species.front();
        for (std::size_t i = 1; i < species.size(); i++) {
            stream << ", " << species[i];
        }
        stream << std::endl;
    }
}

PetscErrorCode ablate::eos::StiffenedGas::StiffenedGasDecodeState(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal *velocity, const PetscReal densityYi[],
                                                                  PetscReal *internalEnergy, PetscReal *a, PetscReal *p, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;

    // Get the velocity in this direction
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        ke += PetscSqr(velocity[d]);
    }
    ke *= 0.5;

    // assumed eos
    (*internalEnergy) = (totalEnergy)-ke;
    *p = (parameters->gamma - 1.0) * density * (*internalEnergy) - parameters->gamma * parameters->p0;
    *a = PetscSqrtReal(parameters->gamma * ((*p) + parameters->p0) / density);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::StiffenedGas::StiffenedGasComputeTemperature(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal *massFlux, const PetscReal densityYi[], PetscReal *T,
                                                                         void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;

    // Get the velocity in this direction
    PetscReal speedSquare = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        speedSquare += PetscSqr(massFlux[d] / density);
    }

    // assumed eos
    PetscReal internalEnergy = (totalEnergy)-0.5 * speedSquare;
    PetscReal cp = parameters->Cp;
    PetscReal pinf = parameters->p0;
    PetscReal gam = parameters->gamma;

    (*T) = (internalEnergy - pinf / density) * gam / cp;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::StiffenedGas::StiffenedGasComputeSpeciesSensibleEnthalpy(PetscReal temperature, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    for (PetscInt s = 0; s < parameters->numberSpecies; s++) {
        hi[s] = 0.0;
    }
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::StiffenedGasComputeDensityFunctionFromTemperaturePressure(PetscReal temperature, PetscReal pressure, const PetscReal *yi, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    PetscReal gam = parameters->gamma;
    *density = (pressure + parameters->p0) / (gam - 1) * gam / parameters->Cp / temperature;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::StiffenedGasComputeSensibleInternalEnergy(PetscReal T, PetscReal density, const PetscReal *yi, PetscReal *sensibleInternalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    PetscReal cp = parameters->Cp;
    PetscReal gam = parameters->gamma;
    PetscReal pinf = parameters->p0;
    *sensibleInternalEnergy = T * cp / gam + pinf / density;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::StiffenedGasComputeSpecificHeatConstantPressure(PetscReal T, PetscReal density, const PetscReal *yi, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    (*specificHeat) = parameters->Cp;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::StiffenedGasComputeSpecificHeatConstantVolume(PetscReal T, PetscReal density, const PetscReal *yi, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    (*specificHeat) = parameters->Cp / parameters->gamma;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::StiffenedGasComputeSensibleEnthalpy(PetscReal T, PetscReal density, const PetscReal *yi, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    // Total Enthalpy == Sensible Enthalpy = e + p/rho
    PetscReal sensibleInternalEnergy;
    PetscErrorCode ierr = ablate::eos::StiffenedGas::StiffenedGasComputeSensibleInternalEnergy(T, density, yi, &sensibleInternalEnergy, ctx);
    CHKERRQ(ierr);

    // Compute the pressure
    Parameters *parameters = (Parameters *)ctx;
    PetscReal p = (parameters->gamma - 1.0) * density * (sensibleInternalEnergy)-parameters->gamma * parameters->p0;

    // compute the enthalpy
    *sensibleEnthalpy = sensibleInternalEnergy + p / density;
    PetscFunctionReturn(0);
}
ablate::eos::ThermodynamicFunction ablate::eos::StiffenedGas::GetThermodynamicFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::PerfectGas requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    return ThermodynamicFunction{.function = thermodynamicFunctions[property].first,
                                 .context = std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2, .eulerOffset = eulerField->offset, .parameters = parameters})};
}
ablate::eos::ThermodynamicTemperatureFunction ablate::eos::StiffenedGas::GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty property,
                                                                                                             const std::vector<domain::Field> &fields) const {
    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::PerfectGas requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    return ThermodynamicTemperatureFunction{
        .function = thermodynamicFunctions[property].second,
        .context = std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2, .eulerOffset = eulerField->offset, .parameters = parameters})};
}

ablate::eos::FieldFunction ablate::eos::StiffenedGas::GetFieldFunctionFunction(const std::string &field, ablate::eos::ThermodynamicProperty property1,
                                                                               ablate::eos::ThermodynamicProperty property2) const {
    return ablate::eos::FieldFunction();
}

PetscErrorCode ablate::eos::StiffenedGas::PressureFunction(const PetscReal *conserved, PetscReal *p, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Get the velocity in this direction
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // assumed eos
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    *p = (functionContext->parameters.gamma - 1.0) * density * (internalEnergy) - functionContext->parameters.gamma * functionContext->parameters.p0;
    //*a = PetscSqrtReal(parameters->gamma * ((*p) + parameters->p0) / density);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::StiffenedGas::PressureTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *p, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal internalEnergy = T*functionContext->parameters.Cp/functionContext->parameters.gamma + functionContext->parameters.p0/density;
    *p = (functionContext->parameters.gamma - 1.0) * density * internalEnergy - functionContext->parameters.gamma * functionContext->parameters.p0;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::StiffenedGas::TemperatureFunction(const PetscReal *conserved, PetscReal *temperature, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Get the velocity in this direction
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal speedSquare = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        speedSquare += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }

    // assumed eos
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - 0.5*speedSquare;
    PetscReal cp = functionContext->parameters.Cp;
    PetscReal pinf = functionContext->parameters.p0;
    PetscReal gam = functionContext->parameters.gamma;

    (*temperature) = (internalEnergy - pinf / density) * gam / cp;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::TemperatureTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *property, void *ctx) {
    return TemperatureFunction(conserved, property,ctx);
}
PetscErrorCode ablate::eos::StiffenedGas::InternalSensibleEnergyFunction(const PetscReal *conserved, PetscReal *internalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Get the velocity in this direction
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal speedSquare = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        speedSquare += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }

    // assumed eos
    *internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - 0.5*speedSquare;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::InternalSensibleEnergyTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *internalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    *internalEnergy = T*functionContext->parameters.Cp/functionContext->parameters.gamma + functionContext->parameters.p0/density;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::SensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *property, void *ctx) { return 0; }
PetscErrorCode ablate::eos::StiffenedGas::SensibleEnthalpyTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *property, void *ctx) { return 0; }
PetscErrorCode ablate::eos::StiffenedGas::SpecificHeatConstantVolumeFunction(const PetscReal *conserved, PetscReal *property, void *ctx) { return 0; }
PetscErrorCode ablate::eos::StiffenedGas::SpecificHeatConstantVolumeTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *property, void *ctx) { return 0; }
PetscErrorCode ablate::eos::StiffenedGas::SpecificHeatConstantPressureFunction(const PetscReal *conserved, PetscReal *property, void *ctx) { return 0; }
PetscErrorCode ablate::eos::StiffenedGas::SpecificHeatConstantPressureTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *property, void *ctx) { return 0; }
PetscErrorCode ablate::eos::StiffenedGas::SpeedOfSoundFunction(const PetscReal *conserved, PetscReal *a, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Get the velocity in this direction
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // assumed eos
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    PetscReal p = (functionContext->parameters.gamma - 1.0) * density * (internalEnergy) - functionContext->parameters.gamma * functionContext->parameters.p0;
    *a = PetscSqrtReal(functionContext->parameters.gamma * (p + functionContext->parameters.p0) / density);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::SpeedOfSoundTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *a, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal internalEnergy = T*functionContext->parameters.Cp/functionContext->parameters.gamma + functionContext->parameters.p0/density;
    PetscReal p = (functionContext->parameters.gamma - 1.0) * density * internalEnergy - functionContext->parameters.gamma * functionContext->parameters.p0;
    *a = PetscSqrtReal(functionContext->parameters.gamma * (p + functionContext->parameters.p0) / density);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::SpeciesSensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *property, void *ctx) { return 0; }
PetscErrorCode ablate::eos::StiffenedGas::SpeciesSensibleEnthalpyTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *property, void *ctx) { return 0; }

#include "registrar.hpp"
REGISTER(ablate::eos::EOS, ablate::eos::StiffenedGas, "stiffened gas eos", ARG(ablate::parameters::Parameters, "parameters", "parameters for the stiffened gas eos"),
         OPT(std::vector<std::string>, "species", "species to track.  Note: species mass fractions do not change eos"));