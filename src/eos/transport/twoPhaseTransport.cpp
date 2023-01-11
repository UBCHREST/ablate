#include "twoPhaseTransport.hpp"
//#include "finiteVolume/processes/twoPhaseEulerAdvection.hpp"
#include "constant.hpp"
#include "sutherland.hpp"

// pass in transport model for each fluid, call in code
ablate::eos::transport::TwoPhaseTransport::TwoPhaseTransport(std::shared_ptr<TransportModel> transportModel1, std::shared_ptr<TransportModel> transportModel2, const std::vector<TransportProperty> &enabledPropertiesIn)
    : transportModel1(std::move(transportModel1)), transportModel2(std::move(transportModel2)),
      enabledProperties(enabledPropertiesIn.empty() ? std::vector<TransportProperty>{TransportProperty::Conductivity, TransportProperty::Viscosity, TransportProperty::Diffusivity} : enabledPropertiesIn) {}

PetscErrorCode ablate::eos::transport::TwoPhaseTransport::TwoPhaseConductivityFunction(const PetscReal *conserved, PetscReal *conductivity, void *ctx) {
    PetscFunctionBeginUser;
    auto contexts = (std::vector<PetscReal> *)ctx;
    // get variable for each of transport model e.g. mu or k or diff
    PetscReal conductivity1, conductivity2;
    conductivity1 = contexts[0][0];
    conductivity2 = contexts[0][1];
    PetscInt vfOffset = contexts[0][4];
    // get alpha from conserved variables
    PetscReal alpha = conserved[vfOffset]; // check index for volumeFraction after pre-stage implementation

    *conductivity = alpha*conductivity1 + (1-alpha)*conductivity2; // for mu and k, not sure about diff
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::TwoPhaseTransport::TwoPhaseConductivityTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *conductivity, void *ctx) {
    PetscFunctionBeginUser;
    // exactly the same as TwoPhaseConductivityFunction
    auto contexts = (std::vector<PetscReal> *)ctx;
    // get variable for each of transport model e.g. mu or k or diff
    PetscReal conductivity1, conductivity2;
    conductivity1 = contexts[0][0];
    conductivity2 = contexts[0][1];
    PetscInt vfOffset = contexts[0][4];
    // get alpha from conserved variables
    PetscReal alpha = conserved[vfOffset]; // check index for volumeFraction after pre-stage implementation

    *conductivity = alpha*conductivity1 + (1-alpha)*conductivity2; // for mu and k, not sure about diff
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::transport::TwoPhaseTransport::TwoPhaseViscosityFunction(const PetscReal *conserved, PetscReal *viscosity, void *ctx) {
    PetscFunctionBeginUser;
    auto contexts = (std::vector<PetscReal> *)ctx;
    // get variable for each of transport model e.g. mu or k or diff
    PetscReal viscosity1, viscosity2;
    viscosity1 = contexts[0][2];
    viscosity2 = contexts[0][3];
    PetscInt vfOffset = contexts[0][4];
    // get alpha from conserved variables
    PetscReal alpha = conserved[vfOffset]; // check index for volumeFraction after pre-stage implementation

    *viscosity = alpha*viscosity1 + (1-alpha)*viscosity2; // for mu and k, not sure about diff
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::TwoPhaseTransport::TwoPhaseViscosityTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *viscosity, void *ctx) {
    PetscFunctionBeginUser;
    auto contexts = (std::vector<PetscReal> *)ctx;
    // get variable for each of transport model e.g. mu or k or diff
    PetscReal viscosity1, viscosity2;
    viscosity1 = contexts[0][2];
    viscosity2 = contexts[0][3];
    PetscInt vfOffset = contexts[0][4];
    // get alpha from conserved variables
    PetscReal alpha = conserved[vfOffset]; // check index for volumeFraction after pre-stage implementation

    *viscosity = alpha*viscosity1 + (1-alpha)*viscosity2; // for mu and k, not sure about diff
    PetscFunctionReturn(0);
}

ablate::eos::ThermodynamicFunction ablate::eos::transport::TwoPhaseTransport::GetTransportFunction(ablate::eos::transport::TransportProperty property, const std::vector<domain::Field> &fields) const {
    if (!std::count(enabledProperties.begin(), enabledProperties.end(), property)) { // check if properties are there
        return ThermodynamicFunction{.function = nullptr, .context = nullptr};
    }
    auto conductivityFunction1 =  this->transportModel1->GetTransportFunction(ablate::eos::transport::TransportProperty::Conductivity, {});
    auto conductivityFunction2 =  this->transportModel2->GetTransportFunction(ablate::eos::transport::TransportProperty::Conductivity, {});
    auto viscosityFunction1 =  this->transportModel1->GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, {});
    auto viscosityFunction2 =  this->transportModel2->GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, {});
    PetscReal k1, k2, mu1, mu2;
    conductivityFunction1.function(nullptr, &k1, conductivityFunction1.context.get());
    conductivityFunction2.function(nullptr, &k2, conductivityFunction2.context.get());
    viscosityFunction1.function(nullptr, &mu1, viscosityFunction1.context.get());
    viscosityFunction2.function(nullptr, &mu2, viscosityFunction2.context.get());
    PetscInt ind, fieldDim;
    fieldDim = fields.size();
    for (PetscInt i=0; i < fieldDim; i++){
        if (fields[i].name == "volumeFraction"){
            ind = i;
        }
    }
    std::vector<PetscReal> contextVec;
    contextVec.resize(5);
    contextVec[0] = k1;
    contextVec[1] = k2;
    contextVec[2] = mu1;
    contextVec[3] = mu2;
    contextVec[4] = fields[ind].offset;

    switch (property) { // not sure about how needs to be changed
        case TransportProperty::Conductivity:
            return ThermodynamicFunction{.function = TwoPhaseConductivityFunction, .context = std::make_shared<std::vector<PetscReal>>(contextVec)};//std::make_shared<ThermodynamicTemperatureFunction>(eos->GetThermodynamicTemperatureFunction)}; // enable share from this (pointer to individual transport models)
        case TransportProperty::Viscosity:
            return ThermodynamicFunction{.function = TwoPhaseViscosityFunction, .context = std::make_shared<std::vector<PetscReal>>(contextVec)};//std::make_shared<double>(mu)};
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
    auto conductivityFunction1 =  this->transportModel1->GetTransportFunction(ablate::eos::transport::TransportProperty::Conductivity, {});
    auto conductivityFunction2 =  this->transportModel2->GetTransportFunction(ablate::eos::transport::TransportProperty::Conductivity, {});
    auto viscosityFunction1 =  this->transportModel1->GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, {});
    auto viscosityFunction2 =  this->transportModel2->GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, {});
    PetscReal k1, k2, mu1, mu2;
    conductivityFunction1.function(nullptr, &k1, conductivityFunction1.context.get());
    conductivityFunction2.function(nullptr, &k2, conductivityFunction2.context.get());
    viscosityFunction1.function(nullptr, &mu1, viscosityFunction1.context.get());
    viscosityFunction2.function(nullptr, &mu2, viscosityFunction2.context.get());
    PetscInt ind, fieldDim;
    fieldDim = fields.size();
    for (PetscInt i=0; i < fieldDim; i++){
        if (fields[i].name == "volumeFraction"){
            ind = i;
        }
    }
    std::vector<PetscReal> contextVec;
    contextVec.resize(5);
    contextVec[0] = k1;
    contextVec[1] = k2;
    contextVec[2] = mu1;
    contextVec[3] = mu2;
    contextVec[4] = fields[ind].offset;


    switch (property) {
        case TransportProperty::Conductivity:
            return ThermodynamicTemperatureFunction{.function = TwoPhaseConductivityTemperatureFunction, .context = std::make_shared<std::vector<PetscReal>>(contextVec)}; // might need different context
        case TransportProperty::Viscosity:
            return ThermodynamicTemperatureFunction{.function = TwoPhaseViscosityTemperatureFunction, .context = std::make_shared<std::vector<PetscReal>>(contextVec)};
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