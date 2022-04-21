#include "constant.hpp"

ablate::eos::transport::Constant::Constant(double k, double mu, double diff) : active((bool)k || (bool)mu || (bool)diff), k(k), mu(mu), diff(diff) {}

PetscErrorCode ablate::eos::transport::Constant::ConstantFunction(const PetscReal *conserved, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    *property = *((double *)ctx);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::Constant::ConstantTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    *property = *((double *)ctx);
    PetscFunctionReturn(0);
}

ablate::eos::ThermodynamicFunction ablate::eos::transport::Constant::GetTransportFunction(ablate::eos::transport::TransportProperty property, const std::vector<domain::Field> &fields) const {
    if (!active) {
        return ThermodynamicFunction{.function = nullptr, .context = nullptr};
    }
    switch (property) {
        case TransportProperty::Conductivity:
            return ThermodynamicFunction{.function = ConstantFunction, .context = std::make_shared<double>(k)};
        case TransportProperty::Viscosity:
            return ThermodynamicFunction{.function = ConstantFunction, .context = std::make_shared<double>(mu)};
        case TransportProperty::Diffusivity:
            return ThermodynamicFunction{.function = ConstantFunction, .context = std::make_shared<double>(diff)};
        default:
            throw std::invalid_argument("Unknown transport property ablate::eos::transport::Constant");
    }
}

ablate::eos::ThermodynamicTemperatureFunction ablate::eos::transport::Constant::GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty property,
                                                                                                                const std::vector<domain::Field> &fields) const {
    if (!active) {
        return ThermodynamicTemperatureFunction{.function = nullptr, .context = nullptr};
    }
    switch (property) {
        case TransportProperty::Conductivity:
            return ThermodynamicTemperatureFunction{.function = ConstantTemperatureFunction, .context = std::make_shared<double>(k)};
        case TransportProperty::Viscosity:
            return ThermodynamicTemperatureFunction{.function = ConstantTemperatureFunction, .context = std::make_shared<double>(mu)};
        case TransportProperty::Diffusivity:
            return ThermodynamicTemperatureFunction{.function = ConstantTemperatureFunction, .context = std::make_shared<double>(diff)};
        default:
            throw std::invalid_argument("Unknown transport property in ablate::eos::transport::Constant");
    }
}

#include "registrar.hpp"
REGISTER_DEFAULT(ablate::eos::transport::TransportModel, ablate::eos::transport::Constant, "constant value transport model (often used for testing)",
                 OPT(double, "k", "thermal conductivity [W/(m K)]"), OPT(double, "mu", "viscosity [Pa s]"), OPT(double, "diff", "diffusivity [m2/s]"));