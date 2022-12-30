#include "sutherland.hpp"

#include <utility>
ablate::eos::transport::Sutherland::Sutherland(std::shared_ptr<eos::EOS> eosIn, const std::vector<TransportProperty> &enabledPropertiesIn)
    : eos(std::move(eosIn)),
      enabledProperties(enabledPropertiesIn.empty() ? std::vector<TransportProperty>{TransportProperty::Conductivity, TransportProperty::Viscosity, TransportProperty::Diffusivity}
                                                    : enabledPropertiesIn) {}

PetscErrorCode ablate::eos::transport::Sutherland::SutherlandConductivityFunction(const PetscReal *conserved, PetscReal *conductivity, void *ctx) {
    PetscFunctionBeginUser;
    const auto &[temperatureFunction, cpFunction] = *(std::pair<ThermodynamicFunction, ThermodynamicTemperatureFunction> *)ctx;
    PetscReal temperature, cp;

    PetscCall(temperatureFunction.function(conserved, &temperature, temperatureFunction.context.get()));
    PetscCall(cpFunction.function(conserved, temperature, &cp, cpFunction.context.get()));

    double mu = muo * PetscSqrtReal(temperature / to) * (temperature / to) * (to + so) / (temperature + so);
    *conductivity = mu * cp / pr;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::Sutherland::SutherlandConductivityTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *conductivity, void *ctx) {
    PetscFunctionBeginUser;
    const auto cpFunction = (ThermodynamicTemperatureFunction *)ctx;
    PetscReal cp;

    PetscCall(cpFunction->function(conserved, temperature, &cp, cpFunction->context.get()));

    double mu = muo * PetscSqrtReal(temperature / to) * (temperature / to) * (to + so) / (temperature + so);
    *conductivity = mu * cp / pr;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::Sutherland::SutherlandViscosityFunction(const PetscReal *conserved, PetscReal *viscosity, void *ctx) {
    PetscFunctionBeginUser;
    const auto temperatureFunction = (ThermodynamicFunction *)ctx;
    PetscReal temperature;

    PetscCall(temperatureFunction->function(conserved, &temperature, temperatureFunction->context.get()));
    *viscosity = muo * PetscSqrtReal(temperature / to) * (temperature / to) * (to + so) / (temperature + so);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::Sutherland::SutherlandViscosityTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *viscosity, void *ctx) {
    PetscFunctionBeginUser;
    *viscosity = muo * PetscSqrtReal(temperature / to) * (temperature / to) * (to + so) / (temperature + so);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::Sutherland::SutherlandDiffusivityFunction(const PetscReal *conserved, PetscReal *diffusivity, void *ctx) {
    PetscFunctionBeginUser;
    const auto &[temperatureFunction, densityFunction] = *(std::pair<ThermodynamicFunction, ThermodynamicFunction> *)ctx;
    PetscReal temperature, density;

    PetscCall(temperatureFunction.function(conserved, &temperature, temperatureFunction.context.get()));
    PetscCall(densityFunction.function(conserved, &density, densityFunction.context.get()));

    double mu = muo * PetscSqrtReal(temperature / to) * (temperature / to) * (to + so) / (temperature + so);
    *diffusivity = mu / density / sc;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::Sutherland::SutherlandDiffusivityTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *diffusivity, void *ctx) {
    PetscFunctionBeginUser;
    const auto densityFunction = (ThermodynamicFunction *)ctx;
    PetscReal density;
    PetscCall(densityFunction->function(conserved, &density, densityFunction->context.get()));
    double mu = muo * PetscSqrtReal(temperature / to) * (temperature / to) * (to + so) / (temperature + so);
    *diffusivity = mu / density / sc;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::Sutherland::SutherlandZeroTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    *property = 0.0;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::transport::Sutherland::SutherlandZeroFunction(const PetscReal *conserved, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    *property = 0.0;
    PetscFunctionReturn(0);
}

ablate::eos::ThermodynamicFunction ablate::eos::transport::Sutherland::GetTransportFunction(ablate::eos::transport::TransportProperty property, const std::vector<domain::Field> &fields) const {
    // check to make sure it is enabled
    if (!std::count(enabledProperties.begin(), enabledProperties.end(), property)) {
        return ThermodynamicFunction{.function = SutherlandZeroFunction, .context = nullptr};
    }

    switch (property) {
        case TransportProperty::Conductivity:
            return ThermodynamicFunction{
                .function = SutherlandConductivityFunction,
                .context = std::make_shared<std::pair<ThermodynamicFunction, ThermodynamicTemperatureFunction>>(
                    eos->GetThermodynamicFunction(ThermodynamicProperty::Temperature, fields), eos->GetThermodynamicTemperatureFunction(ThermodynamicProperty::SpecificHeatConstantPressure, fields))};
        case TransportProperty::Viscosity:
            return ThermodynamicFunction{.function = SutherlandViscosityFunction,
                                         .context = std::make_shared<ThermodynamicFunction>(eos->GetThermodynamicFunction(ThermodynamicProperty::Temperature, fields))};
        case TransportProperty::Diffusivity:
            return ThermodynamicFunction{.function = SutherlandDiffusivityFunction,
                                         .context = std::make_shared<std::pair<ThermodynamicFunction, ThermodynamicFunction>>(eos->GetThermodynamicFunction(ThermodynamicProperty::Temperature, fields),
                                                                                                                              eos->GetThermodynamicFunction(ThermodynamicProperty::Density, fields))};
        default:
            throw std::invalid_argument("Unknown transport property ablate::eos::transport::Constant");
    }
}
ablate::eos::ThermodynamicTemperatureFunction ablate::eos::transport::Sutherland::GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty property,
                                                                                                                  const std::vector<domain::Field> &fields) const {
    // check to make sure it is enabled
    if (!std::count(enabledProperties.begin(), enabledProperties.end(), property)) {
        return ThermodynamicTemperatureFunction{.function = SutherlandZeroTemperatureFunction, .context = nullptr};
    }

    switch (property) {
        case TransportProperty::Conductivity:
            return ThermodynamicTemperatureFunction{
                .function = SutherlandConductivityTemperatureFunction,
                .context = std::make_shared<ThermodynamicTemperatureFunction>(eos->GetThermodynamicTemperatureFunction(ThermodynamicProperty::SpecificHeatConstantPressure, fields))};
        case TransportProperty::Viscosity:
            return ThermodynamicTemperatureFunction{.function = SutherlandViscosityTemperatureFunction, .context = nullptr};
        case TransportProperty::Diffusivity:
            return ThermodynamicTemperatureFunction{.function = SutherlandDiffusivityTemperatureFunction,
                                                    .context = std::make_shared<ThermodynamicFunction>(eos->GetThermodynamicFunction(ThermodynamicProperty::Density, fields))};
        default:
            throw std::invalid_argument("Unknown transport property ablate::eos::transport::Constant");
    }
}

#include "registrar.hpp"
REGISTER(ablate::eos::transport::TransportModel, ablate::eos::transport::Sutherland, "Sutherland Transport model", ARG(ablate::eos::EOS, "eos", "The EOS used to compute Cp (needed for Conductivity)"),
         OPT(std::vector<EnumWrapper<ablate::eos::transport::TransportProperty>>, "enabledProperties", "list of enabled properties.  When empty or default all properties are enabled."));