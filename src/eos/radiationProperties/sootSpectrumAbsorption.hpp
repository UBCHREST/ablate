#ifndef ABLATELIBRARY_SOOTSPECTRUMABSORPTION_HPP
#define ABLATELIBRARY_SOOTSPECTRUMABSORPTION_HPP

#include "eos/tChemSoot.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
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
    };
    const std::shared_ptr<eos::EOS> eos;                                                                               //! eos is needed to compute field values
    constexpr static PetscReal rhoC = 2000;                                                                            // kg/m^3
    constexpr static PetscReal C_2 = (utilities::Constants::h * utilities::Constants::c) / (utilities::Constants::k);  //! Second Plank constant [m K]
    constexpr static PetscReal C_1 =
        2 * ablate::utilities::Constants::pi * ablate::utilities::Constants::h * ablate::utilities::Constants::c * ablate::utilities::Constants::c;  //! First Plank constant [W m^2]
    constexpr static PetscReal C_0 = 7.0;                                                                                                            //! Empirical constant for soot refractive index

    std::vector<PetscReal> wavelengthsIn;

   public:
    SootSpectrumAbsorption(std::shared_ptr<eos::EOS> eosIn, int num = 0, double min = 0.4E-6, double max = 30E-6, const std::vector<double>& wavelengths = {}, const std::vector<double> &bandwidths = {});

    ThermodynamicFunction GetAbsorptionPropertiesFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const;
    ThermodynamicTemperatureFunction GetAbsorptionPropertiesTemperatureFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const;
    static PetscErrorCode SootAbsorptionFunction(const PetscReal* conserved, PetscReal* kappa, void* ctx);
    static PetscErrorCode SootAbsorptionTemperatureFunction(const PetscReal* conserved, PetscReal temperature, PetscReal* kappa, void* ctx);
};
}  // namespace ablate::eos::radiationProperties

#endif  // ABLATELIBRARY_SOOTSPECTRUMABSORPTION_HPP
