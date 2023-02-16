#include "sootSpeciesTransport.hpp"
#include <algorithm>
#include "finiteVolume/compressibleFlowFields.hpp"

ablate::eos::transport::SootSpeciesTransport::SootSpeciesTransport(const std::shared_ptr<TransportModel>& transport) : transport(transport) {}
ablate::eos::ThermodynamicFunction ablate::eos::transport::SootSpeciesTransport::GetTransportFunction(ablate::eos::transport::TransportProperty property,
                                                                                                      const std::vector<domain::Field>& fields) const {
    if (property == ablate::eos::transport::TransportProperty::Diffusivity) {
        // determine the number of species from fields
        auto speciesField = std::find_if(fields.begin(), fields.end(), [](auto field) { return field.name == finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD; });

        if (speciesField == fields.end()) {
            throw std::invalid_argument("ablate::eos::transport::SootSpeciesTransport requires the field finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD");
        }

        auto diffusionFunction = transport->GetTransportFunction(property, fields);
        if (diffusionFunction.propertySize == 1) {
            // update the size
            diffusionFunction.propertySize = speciesField->numberComponents;

            return ThermodynamicFunction{
                .function = VectorizeSpeciesDiffusionFunction, .context = std::make_shared<ThermodynamicFunction>(diffusionFunction), .propertySize = speciesField->numberComponents};
        } else if (diffusionFunction.propertySize == speciesField->numberComponents) {
            return ThermodynamicFunction{
                .function = AdjustSpeciesDiffusionFunction, .context = std::make_shared<ThermodynamicFunction>(diffusionFunction), .propertySize = speciesField->numberComponents};
        } else {
            throw std::invalid_argument("ablate::eos::transport::SootSpeciesTransport requires the transport diffusion function to be sized 1 or number of species");
        }
    } else {
        // just pull from transport
        return transport->GetTransportFunction(property, fields);
    }
}
ablate::eos::ThermodynamicTemperatureFunction ablate::eos::transport::SootSpeciesTransport::GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty property,
                                                                                                                            const std::vector<domain::Field>& fields) const {
    if (property == ablate::eos::transport::TransportProperty::Diffusivity) {
        // determine the number of species from fields
        auto speciesField = std::find_if(fields.begin(), fields.end(), [](auto field) { return field.name == finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD; });

        if (speciesField == fields.end()) {
            throw std::invalid_argument("ablate::eos::transport::SootSpeciesTransport requires the field finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD");
        }

        auto diffusionFunction = transport->GetTransportTemperatureFunction(property, fields);
        if (diffusionFunction.propertySize == 1) {
            // update the size
            diffusionFunction.propertySize = speciesField->numberComponents;

            return ThermodynamicTemperatureFunction{.function = VectorizeSpeciesDiffusionTemperatureFunction,
                                                    .context = std::make_shared<ThermodynamicTemperatureFunction>(diffusionFunction),
                                                    .propertySize = speciesField->numberComponents};
        } else if (diffusionFunction.propertySize == speciesField->numberComponents) {
            return ThermodynamicTemperatureFunction{.function = AdjustSpeciesDiffusionTemperatureFunction,
                                                    .context = std::make_shared<ThermodynamicTemperatureFunction>(diffusionFunction),
                                                    .propertySize = speciesField->numberComponents};
        } else {
            throw std::invalid_argument("ablate::eos::transport::SootSpeciesTransport requires the transport diffusion function to be sized 1 or number of species");
        }
    } else {
        // just pull from transport
        return transport->GetTransportTemperatureFunction(property, fields);
    }
}

PetscErrorCode ablate::eos::transport::SootSpeciesTransport::VectorizeSpeciesDiffusionFunction(const PetscReal* conserved, PetscReal* property, void* ctx) {
    // cast the pointer as a ThermodynamicFunction
    auto functionPointer = static_cast<ThermodynamicFunction*>(ctx);
    PetscReal constantDiff;
    PetscCall(functionPointer->function(conserved, &constantDiff, functionPointer->context.get()));

    // Set all values
    std::fill_n(property, functionPointer->propertySize, constantDiff);

    // Scale the solid carbon
    property[solidCarbonOffset] *= solidCarbonFactor;

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::transport::SootSpeciesTransport::AdjustSpeciesDiffusionFunction(const PetscReal* conserved, PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;

    // cast the pointer as a ThermodynamicFunction
    auto functionPointer = static_cast<ThermodynamicFunction*>(ctx);
    PetscCall(functionPointer->function(conserved, property, functionPointer->context.get()));

    // Scale the solid carbon
    property[solidCarbonOffset] *= solidCarbonFactor;

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::transport::SootSpeciesTransport::VectorizeSpeciesDiffusionTemperatureFunction(const PetscReal* conserved, PetscReal temperature, PetscReal* property, void* ctx) {
    // cast the pointer as a ThermodynamicFunction
    auto functionPointer = static_cast<ThermodynamicTemperatureFunction*>(ctx);
    PetscReal constantDiff;
    PetscCall(functionPointer->function(conserved, temperature, &constantDiff, functionPointer->context.get()));

    // Set all values
    std::fill_n(property, functionPointer->propertySize, constantDiff);

    // Scale the solid carbon
    property[solidCarbonOffset] *= solidCarbonFactor;

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::transport::SootSpeciesTransport::AdjustSpeciesDiffusionTemperatureFunction(const PetscReal* conserved, PetscReal temperature, PetscReal* property, void* ctx) {
    PetscFunctionBeginUser;
    // cast the pointer as a ThermodynamicFunction
    auto functionPointer = static_cast<ThermodynamicTemperatureFunction*>(ctx);
    PetscCall(functionPointer->function(conserved, temperature, property, functionPointer->context.get()));

    // Scale the solid carbon
    property[solidCarbonOffset] *= solidCarbonFactor;

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER_PASS_THROUGH(ablate::eos::transport::TransportModel, ablate::eos::transport::SootSpeciesTransport, "Modifies the transport model for soot", ablate::eos::transport::TransportModel);