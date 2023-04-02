#ifndef ABLATELIBRARY_SOOTSPECIESTRANSPORTMODEL_HPP
#ifndef KOKKOS_ENABLE_CUDA
#define ABLATELIBRARY_SOOTSPECIESTRANSPORTMODEL_HPP

#include <eos/eos.hpp>
#include <memory>
#include "sootTransportModel.hpp"

namespace ablate::eos::tChemSoot {

/**
 * This transport model, reduces the carbon diffusion to be 1% of species diffusion
 */
class SootSpeciesTransportModel : public SootTransportModel {
   public:
    /**
     * This transport model, reduces the carbon diffusion to be 1% of species diffusion
     * @param transport the base transport model
     */
    explicit SootSpeciesTransportModel(const std::shared_ptr<TransportModel>& transportModel);
};
}  // namespace ablate::eos::tChemSoot

#endif  // ABLATELIBRARY_SOOTTRANSPORTMODEL_HPP
#endif