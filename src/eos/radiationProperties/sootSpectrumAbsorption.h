#ifndef ABLATELIBRARY_SOOTSPECTRUMABSORPTION_H
#define ABLATELIBRARY_SOOTSPECTRUMABSORPTION_H

#include "eos/tChemSoot.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "radiationProperties.hpp"

namespace ablate::eos::radiationProperties {
class SootSpectrumAbsorption : public RadiationModel {
   private:
    struct FunctionContext {
        PetscInt densityYiCSolidCOffset;
        const ThermodynamicFunction temperatureFunction;
        const ThermodynamicTemperatureFunction densityFunction;
    };
    const std::shared_ptr<eos::EOS> eos;  //! eos is needed to compute field values
    constexpr static PetscReal rhoC = 2000;  // kg/m^3


   public:
    SootSpectrumAbsorption(std::shared_ptr<EOS> eosIn);

    ThermodynamicFunction GetRadiationPropertiesFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const;
    ThermodynamicTemperatureFunction GetRadiationPropertiesTemperatureFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const;
    static PetscErrorCode SootFunction(const PetscReal* conserved, PetscReal* kappa, void* ctx);
    static PetscErrorCode SootTemperatureFunction(const PetscReal* conserved, PetscReal temperature, PetscReal* kappa, void* ctx);
};
}  // namespace ablate::eos::radiationProperties

#endif  // ABLATELIBRARY_SOOTSPECTRUMABSORPTION_H
