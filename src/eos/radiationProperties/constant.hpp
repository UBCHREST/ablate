#ifndef ABLATELIBRARY_RADIATIONPROPERTIESCONSTANT_H
#define ABLATELIBRARY_RADIATIONPROPERTIESCONSTANT_H

#include "radiation/radiation.hpp"
#include "radiationProperties.hpp"

namespace ablate::eos::radiationProperties {

class Constant : public RadiationModel {
   private:
    struct FunctionContext {
        const PetscReal absorptivity;
        const PetscReal emissivity;
        const ThermodynamicFunction temperatureFunction;
    };

    const std::shared_ptr<eos::EOS> eos;  //! eos is needed to compute temperature
    PetscReal absorptivityIn;
    PetscReal emissivityIn;

    /**
     * Get the refractive index for the material.
     * @return
     */
    static inline constexpr PetscReal GetRefractiveIndex() { return 1; }

    /**
     * private static function for evaluating constant properties without temperature
     * @param conserved
     * @param property
     * @param ctx
     */
    static PetscErrorCode ConstantAbsorptionFunction(const PetscReal conserved[], PetscReal* property, void* ctx);

    /**
     * private static function for evaluating constant properties without temperature
     * @param conserved
     * @param property
     * @param ctx
     */
    static PetscErrorCode ConstantAbsorptionTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);

    /**
     * Returns the black body scaled by the input value of emissivity
     * @param conserved
     * @param property
     * @param ctx
     * @return
     */
    static PetscErrorCode ConstantEmissionFunction(const PetscReal conserved[], PetscReal* property, void* ctx);

    /**
     * Returns the black body scaled by the input value of emissivity
     * @param conserved
     * @param temperature
     * @param property
     * @param ctx
     * @return
     */
    static PetscErrorCode ConstantEmissionTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);

   public:
    explicit Constant(std::shared_ptr<eos::EOS> eosIn, double absorptivity, double emissivity);
    explicit Constant(const Constant&) = delete;
    void operator=(const Constant&) = delete;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields and temperature
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] ThermodynamicTemperatureFunction GetRadiationPropertiesTemperatureFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const override;
};

}  // namespace ablate::eos::radiationProperties

#endif  // ABLATELIBRARY_CONSTANT_H
