#ifndef ABLATELIBRARY_RADIATIONPROPERTIESZIMMER_H
#define ABLATELIBRARY_RADIATIONPROPERTIESZIMMER_H

#include <array>
#include "eos/radiationProperties/radiationProperties.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "radiation/radiation.hpp"
#include "radiationProperties.hpp"
#include "solver/cellSolver.hpp"
#include "utilities/mathUtilities.hpp"

namespace ablate::eos::radiationProperties {
/** A radiation gas absorption model which computes the absorptivity based on the presence of certain species. */
class Zimmer : public RadiationModel {
   private:
    struct FunctionContext {
        PetscInt densityYiH2OOffset;
        PetscInt densityYiCO2Offset;
        PetscInt densityYiCOOffset;
        PetscInt densityYiCH4Offset;

        PetscReal upperLimit;
        PetscReal lowerLimit;

        const ThermodynamicFunction temperatureFunction;
        const ThermodynamicTemperatureFunction densityFunction;
    };

    /**
     * Eos is needed to compute field values
     */
    const std::shared_ptr<eos::EOS> eos;

    /** Some constants that are used in the Zimmer gas absorption model */
    constexpr static double kapparef = 1;  //!< Reference absorptivity
    constexpr static double Tsurf = 300.;  //!< Reference temperature in Kelvin

    /** These are the constants stored for the model. */
    constexpr static std::array<double, 7> H2O_coeff = {0.22317e1, -.15829e1, .1329601e1, -.50707, .93334e-1, -0.83108e-2, 0.28834e-3};
    constexpr static std::array<double, 7> CO2_coeff = {0.38041E1, -0.27808E1, 0.11672E1, -0.284910E0, 0.38163E-1, -0.26292E-2, 0.73662E-4};  //!< polynomial coefficients for H2O
    constexpr static std::array<double, 5> CH4_coeff = {6.6334E0, -3.5686E-3, 1.6682E-8, 2.5611E-10, -2.6558E-14};                            //!< polynomial coefficients for CH4
    constexpr static std::array<double, 5> CO_1_coeff = {4.7869E0, -6.953E-2, 2.95775E-4, -4.25732E-7, 2.02894E-10};                          //!< polynomial coefficients for CO with T <= 750 K
    constexpr static std::array<double, 5> CO_2_coeff = {1.0091E1, -1.183E-2, 4.7753E-6, -5.87209E-10, -2.5334E-14};                          //!< polynomial coefficients for CO with T > 750 K
    constexpr static double MWC = 1.2010700e+01;
    constexpr static double MWO = 1.599940e+01;
    constexpr static double MWH = 1.007940e+00;
    constexpr static double UGC = 8314.4;  //!< Universal Gas Constant J/(K kmol)
    constexpr static double MWCO = MWC + MWO;
    constexpr static double MWCO2 = MWC + 2. * MWO;
    constexpr static double MWCH4 = MWC + 4. * MWH;
    constexpr static double MWH2O = 2. * MWH + MWO;

    /**
     * Returns black body emissivity for the gas
     * @param conserved
     * @param temperature
     * @param epsilon
     * @param ctx
     * @return
     */
    static PetscErrorCode ZimmerEmissionTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal* epsilon, void* ctx);

    /**
     * private static function for evaluating constant properties without temperature
     * @param conserved
     * @param kappa
     * @param ctx
     */
    static PetscErrorCode ZimmerAbsorptionTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal* kappa, void* ctx);

   public:
    explicit Zimmer(std::shared_ptr<eos::EOS> eosIn, PetscReal upperLimitIn = 0, PetscReal lowerLimitIn = 0);
    explicit Zimmer(const Zimmer&) = delete;
    void operator=(const Zimmer&) = delete;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields and temperature
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] ThermodynamicTemperatureFunction GetRadiationPropertiesTemperatureFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const override;

    PetscInt GetFieldComponentOffset(const std::string& str, const domain::Field& field) const;

    PetscReal upperLimitStored;
    PetscReal lowerLimitStored;
};

}  // namespace ablate::eos::radiationProperties

#endif  // ABLATELIBRARY_RADIATIONPROPERTIESZIMMER_H
