#ifndef ABLATELIBRARY_TWOPHASETRANSPORT_HPP
#define ABLATELIBRARY_TWOPHASETRANSPORT_HPP
#include "constant.hpp"
#include "sutherland.hpp"
#include "transportModel.hpp"
// #include "parameters/parameters.hpp"

namespace ablate::eos::transport {

class TwoPhaseTransport : public TransportModel {  //};, std::enable_shared_from_this<TwoPhaseTransport> {
   private:
    const std::shared_ptr<eos::transport::TransportModel> transportModel1;
    const std::shared_ptr<eos::transport::TransportModel> transportModel2;
    struct Contexts {
        PetscReal k1;
        PetscReal k2;
        PetscReal mu1;
        PetscReal mu2;
        PetscReal diff1;  // diffusivity will always be zero for immiscible fluids
        PetscReal diff2;
        PetscInt vfOffset;
    };

    /**
     * Allow specification/disable of certain properties
     **/
    const std::vector<TransportProperty> enabledProperties;

    /** @name Direct and Temperature Based Transport Properties Functions
     * These functions are used to compute the thermodynamic properties.
     * calls.
     * @param conserved
     * @param property
     * @param ctx
     * @return
     * @{
     */
    static PetscErrorCode TwoPhaseConductivityFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode TwoPhaseConductivityTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);
    static PetscErrorCode TwoPhaseViscosityFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode TwoPhaseViscosityTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);
    static PetscErrorCode TwoPhaseDiffusivityFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode TwoPhaseDiffusivityTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);
    static PetscErrorCode TwoPhaseZeroFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode TwoPhaseZeroTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);
    /** @} */

   public:
    explicit TwoPhaseTransport(std::shared_ptr<TransportModel> transportModel1, std::shared_ptr<TransportModel> transportModel2, const std::vector<TransportProperty>& enabledProperties = {});

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

#endif  // ABLATELIBRARY_TWOPHASETRANSPORT_HPP
