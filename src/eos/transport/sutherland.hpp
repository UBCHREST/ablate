#ifndef ABLATELIBRARY_SUTHERLAND_HPP
#define ABLATELIBRARY_SUTHERLAND_HPP

#include <eos/eos.hpp>
#include <memory>
#include "transportModel.hpp"

namespace ablate::eos::transport {

class Sutherland : public TransportModel {
   private:
    /**
     * Eos need to compute cp for sutherland model
     */
    const std::shared_ptr<eos::EOS> eos;

    /**
     * Allow specification/disable of certain properties
     */
    const std::vector<TransportProperty> enabledProperties;

    // constant values
    inline const static PetscReal pr = 0.707;
    inline const static PetscReal muo = 1.716e-5;
    inline const static PetscReal to = 273.e+0;
    inline const static PetscReal so = 111.e+0;
    inline const static PetscReal sc = 0.707;

    /** @name Direct and Temperature Based Transport Properties Functions
     * These functions are used to compute the thermodynamic properties.
     * calls.
     * @param conserved
     * @param property
     * @param ctx
     * @return
     * @{
     */
    static PetscErrorCode SutherlandConductivityFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SutherlandConductivityTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);
    static PetscErrorCode SutherlandViscosityFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SutherlandViscosityTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);
    static PetscErrorCode SutherlandDiffusivityFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SutherlandDiffusivityTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);
    static PetscErrorCode SutherlandZeroFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SutherlandZeroTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);
    /** @} */

   public:
    explicit Sutherland(std::shared_ptr<eos::EOS> eos, const std::vector<TransportProperty>& enabledProperties = {});

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

#endif  // ABLATELIBRARY_SUTHERLAND_HPP
