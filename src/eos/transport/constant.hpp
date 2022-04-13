#ifndef ABLATELIBRARY_TRANSPORT_MODEL_CONSTANT_HPP
#define ABLATELIBRARY_TRANSPORT_MODEL_CONSTANT_HPP
#include "transportModel.hpp"
namespace ablate::eos::transport {

class Constant : public TransportModel {
   private:
    const bool active;
    const PetscReal k;
    const PetscReal mu;
    const PetscReal diff;

    /**
     * private static function for evaluating constant properties without temperature
     * @param conserved
     * @param property
     * @param ctx
     */
    static PetscErrorCode ConstantFunction(const PetscReal conserved[], PetscReal* property, void* ctx);

    /**
     * private static function for evaluating constant properties without temperature
     * @param conserved
     * @param property
     * @param ctx
     */
    static PetscErrorCode ConstantTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);

   public:
    explicit Constant(double k = 0, double mu = 0, double diff = 0);
    explicit Constant(const Constant&) = delete;
    void operator=(const Constant&) = delete;

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
#endif  // ABLATELIBRARY_CONSTANT_HPP
