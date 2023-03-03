#ifndef ABLATELIBRARY_SOOTMEANABSORPTION_HPP
#define ABLATELIBRARY_SOOTMEANABSORPTION_HPP

#include "finiteVolume/compressibleFlowFields.hpp"
#include "radiationProperties.hpp"
#include "utilities/constants.hpp"

namespace ablate::eos::radiationProperties {
/** A radiation soot absorption model which computes the absorptivity of soot based on temperature and number density */
class SootMeanAbsorption : public RadiationModel {
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
    SootMeanAbsorption(std::shared_ptr<EOS> eosIn);
    ThermodynamicFunction GetAbsorptionPropertiesFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const;
    ThermodynamicTemperatureFunction GetAbsorptionPropertiesTemperatureFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const;
    static PetscErrorCode SootFunction(const PetscReal* conserved, PetscReal* kappa, void* ctx);
    static PetscErrorCode SootTemperatureFunction(const PetscReal* conserved, PetscReal temperature, PetscReal* kappa, void* ctx);
};
}  // namespace ablate::eos::radiationProperties
#endif  // ABLATELIBRARY_SOOTMEANABSORPTION_HPP