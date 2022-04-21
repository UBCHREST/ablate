#include "sutherland.hpp"

#include <utility>
ablate::eos::transport::Sutherland::Sutherland(std::shared_ptr<eos::EOS> eosIn) : eos(std::move(eosIn)) {}

PetscErrorCode ablate::eos::transport::Sutherland::SutherlandConductivityFunction(const PetscReal *conserved, PetscReal *conductivity, void *ctx) {
    PetscFunctionBeginUser;
    const auto &[temperatureFunction, cpFunction] = *(std::pair<ThermodynamicFunction, ThermodynamicTemperatureFunction> *)ctx;
    PetscReal temperature, cp;
    PetscErrorCode ierr;

    ierr = temperatureFunction.function(conserved, &temperature, temperatureFunction.context.get());
    CHKERRQ(ierr);
    ierr = cpFunction.function(conserved, temperature, &cp, cpFunction.context.get());
    CHKERRQ(ierr);

    double mu = muo * PetscSqrtReal(temperature / to) * (temperature / to) * (to + so) / (temperature + so);
    *conductivity = mu * cp / pr;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::Sutherland::SutherlandConductivityTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *conductivity, void *ctx) {
    PetscFunctionBeginUser;
    const auto cpFunction = (ThermodynamicTemperatureFunction *)ctx;
    PetscReal cp;
    PetscErrorCode ierr;

    ierr = cpFunction->function(conserved, temperature, &cp, cpFunction->context.get());
    CHKERRQ(ierr);

    double mu = muo * PetscSqrtReal(temperature / to) * (temperature / to) * (to + so) / (temperature + so);
    *conductivity = mu * cp / pr;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::Sutherland::SutherlandViscosityFunction(const PetscReal *conserved, PetscReal *viscosity, void *ctx) {
    PetscFunctionBeginUser;
    const auto temperatureFunction = (ThermodynamicFunction *)ctx;
    PetscReal temperature;
    PetscErrorCode ierr;

    ierr = temperatureFunction->function(conserved, &temperature, temperatureFunction->context.get());
    CHKERRQ(ierr);
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
    PetscErrorCode ierr;

    ierr = temperatureFunction.function(conserved, &temperature, temperatureFunction.context.get());
    CHKERRQ(ierr);
    ierr = densityFunction.function(conserved, &density, densityFunction.context.get());
    CHKERRQ(ierr);

    double mu = muo * PetscSqrtReal(temperature / to) * (temperature / to) * (to + so) / (temperature + so);
    *diffusivity = mu / density / sc;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::Sutherland::SutherlandDiffusivityTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *diffusivity, void *ctx) {
    PetscFunctionBeginUser;
    const auto densityFunction = (ThermodynamicFunction *)ctx;
    PetscReal density;
    PetscErrorCode ierr = densityFunction->function(conserved, &density, densityFunction->context.get());
    CHKERRQ(ierr);
    double mu = muo * PetscSqrtReal(temperature / to) * (temperature / to) * (to + so) / (temperature + so);
    *diffusivity = mu / density / sc;
    PetscFunctionReturn(0);
}
ablate::eos::ThermodynamicFunction ablate::eos::transport::Sutherland::GetTransportFunction(ablate::eos::transport::TransportProperty property, const std::vector<domain::Field> &fields) const {
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
REGISTER(ablate::eos::transport::TransportModel, ablate::eos::transport::Sutherland, "Sutherland Transport model",
         ARG(ablate::eos::EOS, "eos", "The EOS used to compute Cp (needed for Conductivity)"));