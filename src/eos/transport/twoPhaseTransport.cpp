#include "twoPhaseTransport.hpp"

#include "constant.hpp"
#include "sutherland.hpp"

// pass in transport model for each fluid, call in code
ablate::eos::transport::TwoPhaseTransport::TwoPhaseTransport(std::shared_ptr<TransportModel> transportModel1, std::shared_ptr<TransportModel> transportModel2, const std::vector<TransportProperty> &enabledPropertiesIn)
    : transportModel1(std::move(transportModel1)), transportModel2(std::move(transportModel2)),
      enabledProperties(enabledPropertiesIn.empty() ? std::vector<TransportProperty>{TransportProperty::Conductivity, TransportProperty::Viscosity, TransportProperty::Diffusivity} : enabledPropertiesIn) {}

PetscErrorCode ablate::eos::transport::TwoPhaseTransport::TwoPhaseConductivityFunction(const PetscReal *conserved, PetscReal *conductivity, void *ctx) {
    PetscFunctionBeginUser;
    const auto &[conductivityFunction, conductivityTemperatureFunction] = *(std::pair<ThermodynamicFunction, ThermodynamicTemperatureFunction> *)ctx;
    // get variable for each of transport model e.g. mu or k or diff
    PetscReal conductivity1, conductivity2;
    PetscErrorCode ierr;

    ierr = conductivityFunction.function(conserved, &conductivity1, conductivityFunction.context.get());
    CHKERRQ(ierr);
    ierr = conductivityFunction.function(conserved, &conductivity2, conductivityFunction.context.get());
    CHKERRQ(ierr);

    // get alpha from conserved variables
    PetscReal alpha = conserved[0]; // check index for volumeFraction after pre-stage implementation
    //    *property = *((double *)ctx);
    *conductivity = alpha*conductivity1 + (1-alpha)*conductivity2; // for mu and k, not sure about diff
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::TwoPhaseTransport::TwoPhaseConductivityTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *conductivity, void *ctx) {
    PetscFunctionBeginUser;
    // exactly the same as TwoPhaseConductivityFunction
    const auto &[conductivityFunction, conductivityTemperatureFunction] = *(std::pair<ThermodynamicFunction, ThermodynamicTemperatureFunction> *)ctx;
    // get variable for each of transport model e.g. mu or k or diff
    PetscReal conductivity1, conductivity2;
    PetscErrorCode ierr;

    ierr = conductivityFunction.function(conserved, &conductivity1, conductivityFunction.context.get());
    CHKERRQ(ierr);
    ierr = conductivityFunction.function(conserved, &conductivity2, conductivityFunction.context.get());
    CHKERRQ(ierr);

    // get alpha from conserved variables
    PetscReal alpha = conserved[0]; // check index for volumeFraction after pre-stage implementation
    //    *property = *((double *)ctx);
    *conductivity = alpha*conductivity1 + (1-alpha)*conductivity2; // for mu and k, not sure about diff
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::transport::TwoPhaseTransport::TwoPhaseViscosityFunction(const PetscReal *conserved, PetscReal *viscosity, void *ctx) {
    PetscFunctionBeginUser;
    const auto &[viscosityFunction, viscosityTemperatureFunction] = *(std::pair<ThermodynamicFunction, ThermodynamicTemperatureFunction> *)ctx;
    // get variable for each of transport model e.g. mu or k or diff
    PetscReal viscosity1, viscosity2;
    PetscErrorCode ierr;

    ierr = viscosityFunction.function(conserved, &viscosity1, viscosityFunction.context.get());
    CHKERRQ(ierr);
    ierr = viscosityFunction.function(conserved, &viscosity2, viscosityFunction.context.get());
    CHKERRQ(ierr);

    // get alpha from conserved variables
    PetscReal alpha = conserved[0]; // check index for volumeFraction after pre-stage implementation
    //    *property = *((double *)ctx);
    *viscosity = alpha*viscosity1 + (1-alpha)*viscosity2; // for mu and k, not sure about diff
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::TwoPhaseTransport::TwoPhaseViscosityTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *viscosity, void *ctx) {
    PetscFunctionBeginUser;

//    auto context1 = (ablate::eos::transport::Constant *)ctx;
//    PetscReal mu;
//    context1.function(conserved, &mu, context1.context.get());
//    const auto &[viscosityFunction, viscosityTemperatureFunction] = *(std::pair<ThermodynamicFunction, ThermodynamicTemperatureFunction> *)ctx;
    // get variable for each of transport model e.g. mu or k or diff
    PetscReal viscosity1=0, viscosity2=0;
//    viscosity1 = context1->GetTransportFunction(TransportProperty::Conductivity, std::vector<domain::Field>);
//    PetscErrorCode ierr;

//    ierr = viscosityFunction.function(conserved, &viscosity1, viscosityFunction.context.get());
//    CHKERRQ(ierr);
//    ierr = viscosityFunction.function(conserved, &viscosity2, viscosityFunction.context.get());
//    CHKERRQ(ierr);

    // get alpha from conserved variables
    PetscReal alpha = conserved[0]; // check index for volumeFraction after pre-stage implementation
    //    *property = *((double *)ctx);
    *viscosity = alpha*viscosity1 + (1-alpha)*viscosity2; // for mu and k, not sure about diff
    PetscFunctionReturn(0);
}

ablate::eos::ThermodynamicFunction ablate::eos::transport::TwoPhaseTransport::GetTransportFunction(ablate::eos::transport::TransportProperty property, const std::vector<domain::Field> &fields) const {
    if (!std::count(enabledProperties.begin(), enabledProperties.end(), property)) { // check if properties are there
        return ThermodynamicFunction{.function = nullptr, .context = nullptr};
    }
    auto context1 = transportModel1->GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, fields);
    auto context2 = transportModel2->GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, fields);
    switch (property) { // not sure about how needs to be changed
        case TransportProperty::Conductivity:
            return ThermodynamicFunction{.function = TwoPhaseConductivityFunction, .context = nullptr};//std::make_shared<ThermodynamicTemperatureFunction>(eos->GetThermodynamicTemperatureFunction)}; // enable share from this (pointer to individual transport models)
        case TransportProperty::Viscosity:
            return ThermodynamicFunction{.function = TwoPhaseViscosityFunction, .context = (std::make_shared<ThermodynamicFunction>(context1),std::make_shared<ThermodynamicFunction>(context2))};//std::make_shared<double>(mu)};
            //        case TransportProperty::Diffusivity:
            //            return ThermodynamicFunction{.function = TwoPhaseDiffusivityFunction, .context = nullptr};//std::make_shared<double>(diff)};
        default:
            throw std::invalid_argument("Unknown transport property ablate::eos::transport::TwoPhase");
    }
}

ablate::eos::ThermodynamicTemperatureFunction ablate::eos::transport::TwoPhaseTransport::GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty property,
                                                                                                                         const std::vector<domain::Field> &fields) const {
    if (!std::count(enabledProperties.begin(), enabledProperties.end(), property)) {
        return ThermodynamicTemperatureFunction{.function = nullptr, .context = nullptr};
    }
    auto context1 =  transportModel1->GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, fields);

    //    auto context2 = transportModel2;
    switch (property) {
        case TransportProperty::Conductivity:
            return ThermodynamicTemperatureFunction{.function = TwoPhaseConductivityTemperatureFunction, .context = nullptr}; // might need different context
        case TransportProperty::Viscosity:
            return ThermodynamicTemperatureFunction{.function = TwoPhaseViscosityTemperatureFunction, .context = nullptr};
            //        case TransportProperty::Diffusivity:
            //            return ThermodynamicTemperatureFunction{.function = TwoPhaseDiffusivityTemperatureFunction, .context = nullptr};
        default:
            throw std::invalid_argument("Unknown transport property in ablate::eos::transport::TwoPhase");
    }
}

#include "registrar.hpp"
REGISTER(ablate::eos::transport::TransportModel, ablate::eos::transport::TwoPhaseTransport, "transport model for two fluids VOF",
         ARG(ablate::eos::transport::TransportModel, "transport1","Transport model for fluid 1"), ARG(ablate::eos::transport::TransportModel,"transport2","Transport model for fluid 2"),
         OPT(std::vector<EnumWrapper<ablate::eos::transport::TransportProperty>>, "enabledProperties", "list of enabled properties. When empty or default all properties are enabled."));