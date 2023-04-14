#include <Kokkos_Macros.hpp>
#ifndef KOKKOS_ENABLE_CUDA

#include "eos/tChemSoot.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "sootSpeciesTransportModel.hpp"

ablate::eos::tChemSoot::SootSpeciesTransportModel::SootSpeciesTransportModel(const std::shared_ptr<TransportModel>& transportModel)
    : SootTransportModel(transportModel, finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD) {}

ablate::eos::ThermodynamicFunction ablate::eos::tChemSoot::SootSpeciesTransportModel::GetTransportFunction(ablate::eos::transport::TransportProperty property,
                                                                                                           const std::vector<domain::Field>& fields) const {
    if (property == ablate::eos::transport::TransportProperty::Diffusivity) {
        auto speciesField = std::find_if(fields.begin(), fields.end(), [](auto field) { return field.name == finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD; });
        if (speciesField->ComponentIndex(eos::TChemSoot::CSolidName) != 0) {
            throw std::invalid_argument("ablate::eos::tChemSoot::SootTransportModel::SootTransportMode assumes " + eos::TChemSoot::CSolidName + " is the first species.");
        }
    }
    return SootTransportModel::GetTransportFunction(property, fields);
}
ablate::eos::ThermodynamicTemperatureFunction ablate::eos::tChemSoot::SootSpeciesTransportModel::GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty property,
                                                                                                                                 const std::vector<domain::Field>& fields) const {
    if (property == ablate::eos::transport::TransportProperty::Diffusivity) {
        auto speciesField = std::find_if(fields.begin(), fields.end(), [](auto field) { return field.name == finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD; });
        if (speciesField->ComponentIndex(eos::TChemSoot::CSolidName) != 0) {
            throw std::invalid_argument("ablate::eos::tChemSoot::SootTransportModel::SootTransportMode assumes " + eos::TChemSoot::CSolidName + " is the first species.");
        }
    }
    return SootTransportModel::GetTransportTemperatureFunction(property, fields);
}

#include "registrar.hpp"
REGISTER(ablate::eos::transport::TransportModel, ablate::eos::tChemSoot::SootSpeciesTransportModel, "Modifies the transport species model for soot",
         ARG(ablate::eos::transport::TransportModel, "transport", "The baseline transport model.)"));
#endif