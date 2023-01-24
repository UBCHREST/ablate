#ifndef ABLATELIBRARY_TWOPHASETRANSPORT_HPP
#define ABLATELIBRARY_TWOPHASETRANSPORT_HPP

#include "transportModel.hpp"

namespace ablate::eos::transport {

class TwoPhaseTransport : public TransportModel {
   private:
    /**
     * The model corresponding to the volume fraction (alpha)
     */
    const std::shared_ptr<eos::transport::TransportModel> alphaTransportModel;
    /**
     * The model corresponding to (1-alpha)
     */
    const std::shared_ptr<eos::transport::TransportModel> otherTransportModel;

    /**
     * Holds raw pointers to the twoPhase transport function.  It is assumed that the twoPhaseTransport maintains the shared pointer used to generate the function and contextx
     */
    struct FunctionContext {
        // The function for the alpha property
        PetscErrorCode (*alphaFunction)(const PetscReal conserved[], PetscReal* property, void* ctx);
        void* alphaContext;

        // The function for the other property
        PetscErrorCode (*otherFunction)(const PetscReal conserved[], PetscReal* property, void* ctx);
        void* otherContext;

        // offset location for volume fraction in the solution field
        PetscInt vfOffset;
    };

    /**
     * Holds raw pointers to the twoPhase transport temperature function.  It is assumed that the twoPhaseTransport maintains the shared pointer used to generate the function and contextx
     */
    struct TemperatureFunctionContext {
        // The function for the alpha property
        PetscErrorCode (*alphaFunction)(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
        void* alphaContext;

        // The function for the other property
        PetscErrorCode (*otherFunction)(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
        void* otherContext;

        // offset location for volume fraction in the solution field
        PetscInt vfOffset;
    };

    /**
     * Store the list of shared points such that the properties functions remain alive as long as the twoPhaseTransport does
     */
    mutable std::vector<ThermodynamicFunction> thermodynamicFunctionReference;

    /**
     * Store the list of shared points such that the temperature properties functions remain alive as long as the twoPhaseTransport does
     */
    mutable std::vector<ThermodynamicTemperatureFunction> thermodynamicTemperatureFunctionReference;

    /** @name Direct and Temperature Based Transport Properties Functions
     * These functions are used to compute the thermodynamic properties.
     * calls.  They are the same for all properties
     * @param conserved
     * @param property
     * @param ctx
     * @return
     * @{
     */
    static PetscErrorCode TwoPhaseFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode TwoPhaseTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);
    /** @} */

   public:
    explicit TwoPhaseTransport(std::shared_ptr<TransportModel> transportModelAlpha, std::shared_ptr<TransportModel> transportModelOther);

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
