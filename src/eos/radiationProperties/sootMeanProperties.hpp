#ifndef ABLATELIBRARY_SOOTMEANPROPERTIES_HPP
#define ABLATELIBRARY_SOOTMEANPROPERTIES_HPP

#include "finiteVolume/compressibleFlowFields.hpp"
#include "radiationProperties.hpp"
#include "utilities/constants.hpp"
#include "radiation/radiation.hpp"

namespace ablate::eos::radiationProperties {
/** A radiation soot absorption model which computes the absorptivity of soot based on temperature and number density */
class SootMeanProperties : public RadiationModel {
   private:
    struct FunctionContext {
        PetscInt densityYiCSolidCOffset;
        const ThermodynamicFunction temperatureFunction;
        const ThermodynamicTemperatureFunction densityFunction;
    };
    const std::shared_ptr<eos::EOS> eos;  //! eos is needed to compute field values
    constexpr static PetscReal C_2 = (utilities::Constants::h * utilities::Constants::c) / (utilities::Constants::k);
    constexpr static PetscReal C_0 = 7.0;
    constexpr static PetscReal rhoC = 2000;  // kg/m^3
   public:
    SootMeanProperties(std::shared_ptr<EOS> eosIn);
    ThermodynamicFunction GetRadiationPropertiesFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const;
    ThermodynamicTemperatureFunction GetRadiationPropertiesTemperatureFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const;

    static PetscErrorCode SootAbsorptionFunction(const PetscReal* conserved, PetscReal* kappa, void* ctx);
    static PetscErrorCode SootAbsorptionTemperatureFunction(const PetscReal* conserved, PetscReal temperature, PetscReal* kappa, void* ctx);

    static PetscErrorCode SootEmissionFunction(const PetscReal* conserved, PetscReal* epsilon, void* ctx);
    static PetscErrorCode SootEmissionTemperatureFunction(const PetscReal* conserved, PetscReal temperature, PetscReal* epsilon, void* ctx);

    static inline PetscReal GetRefractiveIndex() { return 1; }
};
}  // namespace ablate::eos::radiationProperties
#endif  // ABLATELIBRARY_SOOTMEANPROPERTIES_HPP