#ifndef ABLATELIBRARY_SOOTSPECIESTRANSPORTMODEL_HPP
#define ABLATELIBRARY_SOOTSPECIESTRANSPORTMODEL_HPP
#include <Kokkos_Macros.hpp>
#ifndef KOKKOS_ENABLE_CUDA

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

    /**
     * Single function to produce transport function for any property based upon the available fields
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] ThermodynamicFunction GetTransportFunction(eos::transport::TransportProperty property, const std::vector<domain::Field>& fields) const override;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields and temperature
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] ThermodynamicTemperatureFunction GetTransportTemperatureFunction(eos::transport::TransportProperty property, const std::vector<domain::Field>& fields) const override;
};
}  // namespace ablate::eos::tChemSoot

#endif  // ABLATELIBRARY_SOOTTRANSPORTMODEL_HPP
#endif