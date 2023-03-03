#ifndef ABLATELIBRARY_SOOTSPECTRUMABSORPTION_HPP
#define ABLATELIBRARY_SOOTSPECTRUMABSORPTION_HPP

#include "eos/tChemSoot.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "radiation/radiation.hpp"
#include "radiationProperties.hpp"
#include "utilities/constants.hpp"

namespace ablate::eos::radiationProperties {
class SootSpectrumAbsorption : public RadiationModel {
   private:
    struct FunctionContext {
        PetscInt densityYiCSolidCOffset;
        const ThermodynamicFunction temperatureFunction;
        const ThermodynamicTemperatureFunction densityFunction;
        const std::vector<PetscReal> wavelengths;
        const std::vector<PetscReal> bandwidths;
    };
    const std::shared_ptr<eos::EOS> eos;                                                                               //! eos is needed to compute field values
    constexpr static PetscReal rhoC = 2000;                                                                            // kg/m^3
    constexpr static PetscReal C_2 = (utilities::Constants::h * utilities::Constants::c) / (utilities::Constants::k);  //! Second Plank constant [m K]
    constexpr static PetscReal C_1 =
        2 * ablate::utilities::Constants::pi * ablate::utilities::Constants::h * ablate::utilities::Constants::c * ablate::utilities::Constants::c;  //! First Plank constant [W m^2]
    constexpr static PetscReal C_0 = 7.0;                                                                                                            //! Empirical constant for soot refractive index

    std::vector<PetscReal> wavelengthsIn;
    std::vector<PetscReal> bandwidthsIn;

   public:
    SootSpectrumAbsorption(std::shared_ptr<eos::EOS> eosIn, int num = 0, double min = 0.4E-6, double max = 30E-6, const std::vector<double>& wavelengths = {},
                           const std::vector<double>& bandwidths = {});

    ThermodynamicFunction GetRadiationPropertiesFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const;
    ThermodynamicTemperatureFunction GetRadiationPropertiesTemperatureFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const;

    static PetscErrorCode SootAbsorptionFunction(const PetscReal* conserved, PetscReal* kappa, void* ctx);
    static PetscErrorCode SootAbsorptionTemperatureFunction(const PetscReal* conserved, PetscReal temperature, PetscReal* kappa, void* ctx);

    static PetscErrorCode SootEmissionFunction(const PetscReal* conserved, PetscReal* epsilon, void* ctx);
    static PetscErrorCode SootEmissionTemperatureFunction(const PetscReal* conserved, PetscReal temperature, PetscReal* epsilon, void* ctx);

    //! Polynomial fits to soot data for hydrocarbon combustion conditions (Modest, ch. 11 pg. 432)
    static inline PetscReal GetRefractiveIndex(PetscReal lambda) { return 1.811 + 0.1263 * log(lambda) + 0.027 * log(lambda) * log(lambda) + 0.0417 * log(lambda) * log(lambda) * log(lambda); }
    static inline PetscReal GetAbsorptiveIndex(PetscReal lambda) { return 0.5821 + 0.1213 * log(lambda) + 0.2309 * log(lambda) * log(lambda) - 0.01 * log(lambda) * log(lambda) * log(lambda); }
};
}  // namespace ablate::eos::radiationProperties

#endif  // ABLATELIBRARY_SOOTSPECTRUMABSORPTION_HPP
