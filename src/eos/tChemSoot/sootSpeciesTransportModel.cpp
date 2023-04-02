#ifndef KOKKOS_ENABLE_CUDA
#include "sootSpeciesTransportModel.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"

ablate::eos::tChemSoot::SootSpeciesTransportModel::SootSpeciesTransportModel(const std::shared_ptr<TransportModel>& transportModel)
    : SootTransportModel(transportModel, finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD) {}

#include "registrar.hpp"
REGISTER(ablate::eos::transport::TransportModel, ablate::eos::tChemSoot::SootSpeciesTransportModel, "Modifies the transport species model for soot",
         ARG(ablate::eos::transport::TransportModel, "transport", "The baseline transport model.)"));
#endif