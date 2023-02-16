#include "sootProgressTransportModel.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"

ablate::eos::tChemSoot::SootProgressTransportModel::SootProgressTransportModel(const std::shared_ptr<TransportModel>& transportModel)
    : SootTransportModel(transportModel, finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD) {}

#include "registrar.hpp"
REGISTER_PASS_THROUGH(ablate::eos::tChemSoot::SootTransportModel, ablate::eos::tChemSoot::SootProgressTransportModel, "Modifies the transport progress model for soot",
                      ablate::eos::transport::TransportModel);