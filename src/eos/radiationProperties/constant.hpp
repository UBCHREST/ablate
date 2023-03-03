#ifndef ABLATELIBRARY_RADIATIONPROPERTIESCONSTANT_H
#define ABLATELIBRARY_RADIATIONPROPERTIESCONSTANT_H

#include "radiationProperties.hpp"

namespace ablate::eos::radiationProperties {

class Constant : public RadiationModel {
   private:
    const PetscReal absorptivity;

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
    explicit Constant(double absorptivity);
    explicit Constant(const Constant&) = delete;
    void operator=(const Constant&) = delete;

    /**
     * Single function to produce radiation properties function for any property based upon the available fields
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] ThermodynamicFunction GetAbsorptionPropertiesFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const override;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields and temperature
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] ThermodynamicTemperatureFunction GetAbsorptionPropertiesTemperatureFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const override;
};

}  // namespace ablate::eos::radiationProperties

#endif  // ABLATELIBRARY_CONSTANT_H
