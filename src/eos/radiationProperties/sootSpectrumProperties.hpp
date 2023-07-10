#ifndef ABLATELIBRARY_SOOTSPECTRUMPROPERTIES_HPP
#define ABLATELIBRARY_SOOTSPECTRUMPROPERTIES_HPP

#include "eos/tChemSoot.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "radiation/radiation.hpp"
#include "radiationProperties.hpp"
#include "utilities/constants.hpp"

namespace ablate::eos::radiationProperties {
class SootSpectrumProperties : public RadiationModel {
   private:
    struct FunctionContext {
        PetscInt densityYiCSolidCOffset;
        const ThermodynamicFunction temperatureFunction;
        const ThermodynamicTemperatureFunction densityFunction;
        const std::vector<PetscReal> wavelengths;
        const std::vector<PetscReal> bandwidths;
    };
    const std::shared_ptr<eos::EOS> eos;     //! eos is needed to compute field values
    constexpr static PetscReal rhoC = 2000;  //! kg/m^3

    std::vector<PetscReal> wavelengthsIn;
    std::vector<PetscReal> bandwidthsIn;

   public:
    SootSpectrumProperties(std::shared_ptr<eos::EOS> eosIn, int num = 0, double min = 0.4E-6, double max = 30E-6, const std::vector<double>& wavelengths = {},
                           const std::vector<double>& bandwidths = {});

    ThermodynamicFunction GetRadiationPropertiesFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const;
    ThermodynamicTemperatureFunction GetRadiationPropertiesTemperatureFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const;

    static PetscErrorCode SootAbsorptionTemperatureFunction(const PetscReal* conserved, PetscReal temperature, PetscReal* kappa, void* ctx);

    static PetscErrorCode SootEmissionTemperatureFunction(const PetscReal* conserved, PetscReal temperature, PetscReal* epsilon, void* ctx);

    //! Polynomial fits to soot data for hydrocarbon combustion conditions (Modest, ch. 11 pg. 432)
    static inline PetscReal GetRefractiveIndex(PetscReal lambda) {
        lambda *= 1E6;
        return 1.811 + (0.1263 * log(lambda)) + (0.027 * log(lambda) * log(lambda)) + (0.0417 * log(lambda) * log(lambda) * log(lambda));
    }
    static inline PetscReal GetAbsorptiveIndex(PetscReal lambda) {
        lambda *= 1E6;
        return 0.5821 + (0.1213 * log(lambda)) + (0.2309 * log(lambda) * log(lambda)) - (0.01 * log(lambda) * log(lambda) * log(lambda));
    }
};
}  // namespace ablate::eos::radiationProperties

#endif  // ABLATELIBRARY_SOOTSPECTRUMPROPERTIES_HPP
