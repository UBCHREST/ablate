#include "sootSpeciesTransportModel.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"

ablate::eos::tChemSoot::SootSpeciesTransportModel::SootSpeciesTransportModel(const std::shared_ptr<TransportModel>& transportModel)
    : SootTransportModel(transportModel, finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD) {}

#include "registrar.hpp"
REGISTER_PASS_THROUGH(ablate::eos::tChemSoot::SootTransportModel, ablate::eos::tChemSoot::SootSpeciesTransportModel, "Modifies the transport species model for soot",
                      ablate::eos::transport::TransportModel);