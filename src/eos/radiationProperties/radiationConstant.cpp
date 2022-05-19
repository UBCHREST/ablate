//
// Created by owen on 5/19/22.
//
#include "radiationConstant.hpp"

ablate::eos::radiationProperties::Constant::Constant(double absorptivity) : active((bool)absorptivity), absorptivity(absorptivity){}

PetscErrorCode ablate::eos::radiationProperties::Constant::ConstantFunction(const PetscReal *conserved, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    *property = *((double *)ctx);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::radiationProperties::Constant::ConstantTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    *property = *((double *)ctx);
    PetscFunctionReturn(0);
}

ablate::eos::ThermodynamicFunction ablate::eos::radiationProperties::Constant::GetRadiationPropertiesFunction(radiationModel::RadiationProperty property, const std::vector<domain::Field> &fields) const {
    if (!active) {
        return ThermodynamicFunction{.function = nullptr, .context = nullptr};
    }
    switch (property) {
        case radiationModel::RadiationProperty::Absorptivity:
            return ThermodynamicFunction{.function = ConstantFunction, .context = std::make_shared<double>(absorptivity)};
        default:
            throw std::invalid_argument("Unknown radiationProperties property ablate::eos::radiationProperties::Constant");
    }
}

ablate::eos::ThermodynamicTemperatureFunction ablate::eos::radiationProperties::Constant::GetRadiationPropertiesTemperatureFunction(radiationModel::RadiationProperty property,
                                                                                                                const std::vector<domain::Field> &fields) const {
    if (!active) {
        return ThermodynamicTemperatureFunction{.function = nullptr, .context = nullptr};
    }
    switch (property) {
        case radiationModel::RadiationProperty::Absorptivity:
            return ThermodynamicTemperatureFunction{.function = ConstantTemperatureFunction, .context = std::make_shared<double>(absorptivity)};
        default:
            throw std::invalid_argument("Unknown radiationProperties property in ablate::eos::radiationProperties::Constant");
    }
}