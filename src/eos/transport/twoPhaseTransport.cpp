#include "twoPhaseTransport.hpp"
#include "constant.hpp"
#include "sutherland.hpp"

// pass in transport model for each fluid, call in code
ablate::eos::transport::TwoPhaseTransport::TwoPhaseTransport(std::shared_ptr<TransportModel> transportModel1, std::shared_ptr<TransportModel> transportModel2,
                                                             const std::vector<TransportProperty> &enabledPropertiesIn)
    : transportModel1(std::move(transportModel1)),
      transportModel2(std::move(transportModel2)),
      enabledProperties(enabledPropertiesIn.empty() ? std::vector<TransportProperty>{TransportProperty::Conductivity, TransportProperty::Viscosity, TransportProperty::Diffusivity}
                                                    : enabledPropertiesIn) {}

PetscErrorCode ablate::eos::transport::TwoPhaseTransport::TwoPhaseConductivityFunction(const PetscReal *conserved, PetscReal *conductivity, void *ctx) {
    PetscFunctionBeginUser;
    auto contexts = (struct Contexts *)ctx;
    // get k for each transport model
    PetscReal conductivity1, conductivity2;
    conductivity1 = contexts->k1;
    conductivity2 = contexts->k2;
    PetscInt vfOffset = contexts->vfOffset;
    // get alpha from conserved variables
    PetscReal alpha = conserved[vfOffset];

    *conductivity = alpha * conductivity1 + (1 - alpha) * conductivity2;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::TwoPhaseTransport::TwoPhaseConductivityTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *conductivity, void *ctx) {
    PetscFunctionBeginUser;
    // exactly the same as TwoPhaseConductivityFunction
    auto contexts = (struct Contexts *)ctx;
    // get k for each transport model
    PetscReal conductivity1, conductivity2;
    conductivity1 = contexts->k1;
    conductivity2 = contexts->k2;
    PetscInt vfOffset = contexts->vfOffset;
    // get alpha from conserved variables
    PetscReal alpha = conserved[vfOffset];

    *conductivity = alpha * conductivity1 + (1 - alpha) * conductivity2;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::transport::TwoPhaseTransport::TwoPhaseViscosityFunction(const PetscReal *conserved, PetscReal *viscosity, void *ctx) {
    PetscFunctionBeginUser;
    auto contexts = (struct Contexts *)ctx;
    // get mu for each transport model
    PetscReal viscosity1, viscosity2;
    viscosity1 = contexts->mu1;
    viscosity2 = contexts->mu2;
    PetscInt vfOffset = contexts->vfOffset;
    // get alpha from conserved variables
    PetscReal alpha = conserved[vfOffset];

    *viscosity = alpha * viscosity1 + (1 - alpha) * viscosity2;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::TwoPhaseTransport::TwoPhaseViscosityTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *viscosity, void *ctx) {
    PetscFunctionBeginUser;
    auto contexts = (struct Contexts *)ctx;
    // get mu for each transport model
    PetscReal viscosity1, viscosity2;
    viscosity1 = contexts->mu1;
    viscosity2 = contexts->mu2;
    PetscInt vfOffset = contexts->vfOffset;
    // get alpha from conserved variables
    PetscReal alpha = conserved[vfOffset];

    *viscosity = alpha * viscosity1 + (1 - alpha) * viscosity2;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::transport::TwoPhaseTransport::TwoPhaseDiffusivityFunction(const PetscReal *conserved, PetscReal *diffusivity, void *ctx) {
    PetscFunctionBeginUser;
    *diffusivity = 0.0;  // not sure if this is correct. diffusivity between immiscible two phase should be zero?
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::TwoPhaseTransport::TwoPhaseDiffusivityTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *diffusivity, void *ctx) {
    PetscFunctionBeginUser;
    *diffusivity = 0.0;  // not sure if this is correct. diffusivity between immiscible two phase should be zero?
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::TwoPhaseTransport::TwoPhaseZeroFunction(const PetscReal *conserved, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    *property = 0.0;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::TwoPhaseTransport::TwoPhaseZeroTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    *property = 0.0;
    PetscFunctionReturn(0);
}

ablate::eos::ThermodynamicFunction ablate::eos::transport::TwoPhaseTransport::GetTransportFunction(ablate::eos::transport::TransportProperty property, const std::vector<domain::Field> &fields) const {
    if (!std::count(enabledProperties.begin(), enabledProperties.end(), property)) {  // check if properties are there
        return ThermodynamicFunction{.function = TwoPhaseZeroFunction, .context = nullptr};
    }
    auto conductivityFunction1 = this->transportModel1->GetTransportFunction(ablate::eos::transport::TransportProperty::Conductivity, {});
    auto conductivityFunction2 = this->transportModel2->GetTransportFunction(ablate::eos::transport::TransportProperty::Conductivity, {});
    auto viscosityFunction1 = this->transportModel1->GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, {});
    auto viscosityFunction2 = this->transportModel2->GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, {});
    PetscReal k1, k2, mu1, mu2;
    conductivityFunction1.function(nullptr, &k1, conductivityFunction1.context.get());
    conductivityFunction2.function(nullptr, &k2, conductivityFunction2.context.get());
    viscosityFunction1.function(nullptr, &mu1, viscosityFunction1.context.get());
    viscosityFunction2.function(nullptr, &mu2, viscosityFunction2.context.get());
    PetscInt ind, fieldDim;
    fieldDim = fields.size();
    Contexts contexts;

    if (fieldDim == 0) {
        contexts.vfOffset = 0;  // no fields passed in, volume fraction is only field
    } else {
        for (PetscInt i = 0; i < fieldDim; i++) {
            if (fields[i].name == "volumeFraction") {
                ind = i;
            }
        }
        contexts.vfOffset = fields[ind].offset;
    }

    contexts.k1 = k1;
    contexts.k2 = k2;
    contexts.mu1 = mu1;
    contexts.mu2 = mu2;

    switch (property) {
        case TransportProperty::Conductivity:
            return ThermodynamicFunction{.function = TwoPhaseConductivityFunction, .context = std::make_shared<struct Contexts>(contexts)};
        case TransportProperty::Viscosity:
            return ThermodynamicFunction{.function = TwoPhaseViscosityFunction, .context = std::make_shared<struct Contexts>(contexts)};
        case TransportProperty::Diffusivity:
            return ThermodynamicFunction{.function = TwoPhaseDiffusivityFunction, .context = nullptr};
        default:
            throw std::invalid_argument("Unknown transport property ablate::eos::transport::TwoPhase");
    }
}

ablate::eos::ThermodynamicTemperatureFunction ablate::eos::transport::TwoPhaseTransport::GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty property,
                                                                                                                         const std::vector<domain::Field> &fields) const {
    // check to make sure it is enabled
    if (!std::count(enabledProperties.begin(), enabledProperties.end(), property)) {
        return ThermodynamicTemperatureFunction{.function = TwoPhaseZeroTemperatureFunction, .context = nullptr};
    }
    auto conductivityFunction1 = this->transportModel1->GetTransportFunction(ablate::eos::transport::TransportProperty::Conductivity, {});
    auto conductivityFunction2 = this->transportModel2->GetTransportFunction(ablate::eos::transport::TransportProperty::Conductivity, {});
    auto viscosityFunction1 = this->transportModel1->GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, {});
    auto viscosityFunction2 = this->transportModel2->GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, {});
    PetscReal k1, k2, mu1, mu2;
    conductivityFunction1.function(nullptr, &k1, conductivityFunction1.context.get());
    conductivityFunction2.function(nullptr, &k2, conductivityFunction2.context.get());
    viscosityFunction1.function(nullptr, &mu1, viscosityFunction1.context.get());
    viscosityFunction2.function(nullptr, &mu2, viscosityFunction2.context.get());
    PetscInt ind, fieldDim;
    fieldDim = fields.size();
    Contexts contexts;

    if (fieldDim == 0) {
        contexts.vfOffset = 0;
    } else {
        for (PetscInt i = 0; i < fieldDim; i++) {
            if (fields[i].name == "volumeFraction") {
                ind = i;
            }
        }
        contexts.vfOffset = fields[ind].offset;
    }

    contexts.k1 = k1;
    contexts.k2 = k2;
    contexts.mu1 = mu1;
    contexts.mu2 = mu2;

    switch (property) {
        case TransportProperty::Conductivity:
            return ThermodynamicTemperatureFunction{.function = TwoPhaseConductivityTemperatureFunction, .context = std::make_shared<struct Contexts>(contexts)};
        case TransportProperty::Viscosity:
            return ThermodynamicTemperatureFunction{.function = TwoPhaseViscosityTemperatureFunction, .context = std::make_shared<struct Contexts>(contexts)};
        case TransportProperty::Diffusivity:
            return ThermodynamicTemperatureFunction{.function = TwoPhaseDiffusivityTemperatureFunction, .context = nullptr};
        default:
            throw std::invalid_argument("Unknown transport property in ablate::eos::transport::TwoPhase");
    }
}

#include "registrar.hpp"
REGISTER(ablate::eos::transport::TransportModel, ablate::eos::transport::TwoPhaseTransport, "transport model for two fluids VOF",
         ARG(ablate::eos::transport::TransportModel, "transport1", "Transport model for fluid 1"), ARG(ablate::eos::transport::TransportModel, "transport2", "Transport model for fluid 2"),
         OPT(std::vector<EnumWrapper<ablate::eos::transport::TransportProperty>>, "enabledProperties", "list of enabled properties. When empty or default all properties are enabled."));