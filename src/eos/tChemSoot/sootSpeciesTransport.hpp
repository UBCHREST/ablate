#ifndef ABLATELIBRARY_SOOTSPECIESPTRANSPORT_HPP
#define ABLATELIBRARY_SOOTSPECIESPTRANSPORT_HPP

#include <eos/eos.hpp>
#include <memory>
#include "eos/transport/transportModel.hpp"

namespace ablate::eos::transport {

/**
 * This transport model, reduces the carbon diffusion to be 1% of species diffusion
 */
class SootSpeciesTransport : public TransportModel {
   private:
    /**
     * The baseline transport used for conductivity, viscosity, and baseline species diffusion
     */
    const std::shared_ptr<TransportModel> transport;

    /**
     * Scale the solid carbon diffusion by a constant value
     */
    constexpr static PetscReal solidCarbonFactor = 1.0 / 100.0;

    /**
     * Store the solid carbon offset, TChem soot makes this always zero
     */
    constexpr static std::size_t solidCarbonOffset = 0;

    /**
     * private static function to vectorize the species diffusion and set the solidCarbonOffset value to small
     * @param conserved
     * @param property
     * @param ctx
     */
    static PetscErrorCode VectorizeSpeciesDiffusionFunction(const PetscReal conserved[], PetscReal* property, void* ctx);

    /**
     * private static function use a vectorized species diffusion and set the solidCarbonOffset value to small
     * @param conserved
     * @param property
     * @param ctx
     */
    static PetscErrorCode AdjustSpeciesDiffusionFunction(const PetscReal conserved[], PetscReal* property, void* ctx);

    /**
     * private static function to vectorize the species diffusion and set the solidCarbonOffset value to small
     * @param conserved
     * @param property
     * @param ctx
     */
    static PetscErrorCode VectorizeSpeciesDiffusionTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);


    /**
     * private static function use a vectorized species diffusion and set the solidCarbonOffset value to small
     * @param conserved
     * @param property
     * @param ctx
     */
    static PetscErrorCode AdjustSpeciesDiffusionTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);


   public:
    /**
     * This transport model, reduces the carbon diffusion to be 1% of species diffusion
     * @param eos the eos to use for the species list
     * @param transport the base transport model
     */
    explicit SootSpeciesTransport(const std::shared_ptr<TransportModel>& transport);

    /**
     * Single function to produce transport function for any property based upon the available fields
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] ThermodynamicFunction GetTransportFunction(TransportProperty property, const std::vector<domain::Field>& fields) const override;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields and temperature
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] ThermodynamicTemperatureFunction GetTransportTemperatureFunction(TransportProperty property, const std::vector<domain::Field>& fields) const override;
};
}  // namespace ablate::eos::transport

#endif  // ABLATELIBRARY_SOOTSPECIESPTRANSPORT_HPP
