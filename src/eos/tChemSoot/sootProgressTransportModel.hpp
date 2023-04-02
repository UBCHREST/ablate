#ifndef ABLATELIBRARY_SOOTPROGRESSTRANSPORTMODEL_HPP
#ifndef KOKKOS_ENABLE_CUDA
#define ABLATELIBRARY_SOOTPROGRESSTRANSPORTMODEL_HPP

#include <eos/eos.hpp>
#include <memory>
#include "sootTransportModel.hpp"

namespace ablate::eos::tChemSoot {

/**
 * This transport model, reduces the NDD diffusion to be 1% of bulkd diffusion
 */
class SootProgressTransportModel : public SootTransportModel {
   public:
    /**
     * This transport model, reduces the carbon diffusion to be 1% of buld diffusion
     * @param transport the base transport model
     */
    explicit SootProgressTransportModel(const std::shared_ptr<TransportModel>& transportModel);
};
}  // namespace ablate::eos::tChemSoot

#endif  // ABLATELIBRARY_SOOTPROGRESSTRANSPORTMODEL_HPP
#endif