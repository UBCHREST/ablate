#include <Kokkos_Macros.hpp>
#ifndef KOKKOS_ENABLE_CUDA

#include "finiteVolume/compressibleFlowFields.hpp"
#include "sootProgressTransportModel.hpp"

ablate::eos::tChemSoot::SootProgressTransportModel::SootProgressTransportModel(const std::shared_ptr<TransportModel>& transportModel)
    : SootTransportModel(transportModel, finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD) {}

#include "registrar.hpp"
REGISTER(ablate::eos::transport::TransportModel, ablate::eos::tChemSoot::SootProgressTransportModel, "Modifies the transport progress model for soot",
         ARG(ablate::eos::transport::TransportModel, "transport", "The baseline transport model.)"));
#endif