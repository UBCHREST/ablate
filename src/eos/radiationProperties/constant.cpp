#include "constant.hpp"

ablate::eos::radiationProperties::Constant::Constant(double absorptivity) : absorptivity(absorptivity) {}

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

ablate::eos::ThermodynamicFunction ablate::eos::radiationProperties::Constant::GetRadiationPropertiesFunction(RadiationProperty property, const std::vector<domain::Field> &fields) const {
    switch (property) {
        case RadiationProperty::Absorptivity:
            return ThermodynamicFunction{.function = ConstantFunction, .context = std::make_shared<double>(absorptivity)};
        default:
            throw std::invalid_argument("Unknown radiationProperties property ablate::eos::radiationProperties::Constant");
    }
}

ablate::eos::ThermodynamicTemperatureFunction ablate::eos::radiationProperties::Constant::GetRadiationPropertiesTemperatureFunction(RadiationProperty property,
                                                                                                                                    const std::vector<domain::Field> &fields) const {
    switch (property) {
        case RadiationProperty::Absorptivity:
            return ThermodynamicTemperatureFunction{.function = ConstantTemperatureFunction, .context = std::make_shared<double>(absorptivity)};
        default:
            throw std::invalid_argument("Unknown radiationProperties property in ablate::eos::radiationProperties::Constant");
    }
}

#include "registrar.hpp"
REGISTER(ablate::eos::radiationProperties::RadiationModel, ablate::eos::radiationProperties::Constant, "constant value transport model (often used for testing)",
         ARG(double, "absorptivity", "radiative absorptivity"));