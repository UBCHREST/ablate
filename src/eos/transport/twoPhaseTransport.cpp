#include "twoPhaseTransport.hpp"
#include "finiteVolume/processes/twoPhaseEulerAdvection.hpp"

ablate::eos::transport::TwoPhaseTransport::TwoPhaseTransport(std::shared_ptr<TransportModel> alphaTransportModel, std::shared_ptr<TransportModel> otherTransportModel)
    : alphaTransportModel(std::move(alphaTransportModel)), otherTransportModel(std::move(otherTransportModel)) {}

PetscErrorCode ablate::eos::transport::TwoPhaseTransport::TwoPhaseFunction(const PetscReal *conserved, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    auto contexts = (FunctionContext *)ctx;
    // get property for each transport model
    PetscReal propertyAlpha, propertyOther;

    // Call the eval for each property
    PetscCall(contexts->alphaFunction(conserved, &propertyAlpha, contexts->alphaContext));
    PetscCall(contexts->otherFunction(conserved, &propertyOther, contexts->otherContext));

    // get alpha from conserved variables
    PetscInt vfOffset = contexts->vfOffset;
    PetscReal alpha = conserved[vfOffset];

    *property = alpha * propertyAlpha + (1.0 - alpha) * propertyOther;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::TwoPhaseTransport::TwoPhaseTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    auto contexts = (TemperatureFunctionContext *)ctx;
    // get property for each transport model
    PetscReal propertyAlpha, propertyOther;

    // Call the eval for each property
    PetscCall(contexts->alphaFunction(conserved, temperature, &propertyAlpha, contexts->alphaContext));
    PetscCall(contexts->otherFunction(conserved, temperature, &propertyOther, contexts->otherContext));

    // get alpha from conserved variables
    PetscInt vfOffset = contexts->vfOffset;
    PetscReal alpha = conserved[vfOffset];

    *property = alpha * propertyAlpha + (1.0 - alpha) * propertyOther;
    PetscFunctionReturn(0);
}

ablate::eos::ThermodynamicFunction ablate::eos::transport::TwoPhaseTransport::GetTransportFunction(ablate::eos::transport::TransportProperty property, const std::vector<domain::Field> &fields) const {
    auto alphaFunction = alphaTransportModel->GetTransportFunction(property, fields);
    auto otherFunction = otherTransportModel->GetTransportFunction(property, fields);

    // If either function is null, return a null function
    if (!(alphaFunction.function && otherFunction.function)) {
        return ThermodynamicFunction{.function = nullptr, .context = nullptr};
    }

    // Store the functions to keep the reference count up
    thermodynamicFunctionReference.push_back(alphaFunction);
    thermodynamicFunctionReference.push_back(otherFunction);

    // Find the volumeFraction field
    const auto volumeFractionField =
        std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD; });
    if (volumeFractionField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::transport::TwoPhaseTransport requires the ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD Field");
    }

    FunctionContext context{.alphaFunction = alphaFunction.function,
                            .alphaContext = alphaFunction.context.get(),
                            .otherFunction = otherFunction.function,
                            .otherContext = otherFunction.context.get(),
                            .vfOffset = volumeFractionField->offset};

    return ThermodynamicFunction{.function = TwoPhaseFunction, .context = std::make_shared<FunctionContext>(context)};
}

ablate::eos::ThermodynamicTemperatureFunction ablate::eos::transport::TwoPhaseTransport::GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty property,
                                                                                                                         const std::vector<domain::Field> &fields) const {
    auto alphaFunction = alphaTransportModel->GetTransportTemperatureFunction(property, fields);
    auto otherFunction = otherTransportModel->GetTransportTemperatureFunction(property, fields);

    // If either function is null, return a null function
    if (!(alphaFunction.function && otherFunction.function)) {
        return ThermodynamicTemperatureFunction{.function = nullptr, .context = nullptr};
    }

    // Store the functions to keep the reference count up
    thermodynamicTemperatureFunctionReference.push_back(alphaFunction);
    thermodynamicTemperatureFunctionReference.push_back(otherFunction);

    // Find the volumeFraction field
    const auto volumeFractionField =
        std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD; });
    if (volumeFractionField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::transport::TwoPhaseTransport requires the ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD Field");
    }

    TemperatureFunctionContext context{.alphaFunction = alphaFunction.function,
                                       .alphaContext = alphaFunction.context.get(),
                                       .otherFunction = otherFunction.function,
                                       .otherContext = otherFunction.context.get(),
                                       .vfOffset = volumeFractionField->offset};

    return ThermodynamicTemperatureFunction{.function = TwoPhaseTemperatureFunction, .context = std::make_shared<TemperatureFunctionContext>(context)};
}

#include "registrar.hpp"
REGISTER(ablate::eos::transport::TransportModel, ablate::eos::transport::TwoPhaseTransport, "transport model for two fluids VOF",
         ARG(ablate::eos::transport::TransportModel, "transport1", "Transport model for fluid 1 (vf)"),
         ARG(ablate::eos::transport::TransportModel, "transport2", "Transport model for fluid 2 (1-vf)"));