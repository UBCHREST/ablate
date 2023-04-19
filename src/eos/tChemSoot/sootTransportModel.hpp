#ifndef ABLATELIBRARY_SOOTTRANSPORTMODEL_HPP
#define ABLATELIBRARY_SOOTTRANSPORTMODEL_HPP
#include <Kokkos_Macros.hpp>
#ifndef KOKKOS_ENABLE_CUDA

#include <eos/eos.hpp>
#include <memory>
#include "eos/transport/transportModel.hpp"

namespace ablate::eos::tChemSoot {

/**
 * This transport model, reduces the carbon diffusion to be 1% of species diffusion
 */
class SootTransportModel : public eos::transport::TransportModel {
   private:
    //! Store the name of the field to find to set to zero
    const std::string fieldName;
    /**
     * The baseline transport used for conductivity, viscosity, and baseline species diffusion
     */
    const std::shared_ptr<TransportModel> transport;

    /**
     * Scale the solid carbon diffusion by a constant value
     */
    constexpr static PetscReal solidCarbonFactor = 1.0 / 100.0;

    /**
     * Store the solid carbon/ndd offset, TChem soot makes this always zero
     */
    constexpr static std::size_t offset = 0;

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
     * @param transport the base transport model
     * @param componentName, the name of the component to adjust (solid carbon or ndd)
     */
    explicit SootTransportModel(const std::shared_ptr<TransportModel>& transport, std::string componentName);

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