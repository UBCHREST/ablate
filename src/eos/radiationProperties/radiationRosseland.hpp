#ifndef ABLATELIBRARY_RADIATIONPROPERTIESROSSELAND_H
#define ABLATELIBRARY_RADIATIONPROPERTIESROSSELAND_H

#include "eos/radiationProperties/radiationProperties.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "radiationProperties.hpp"
#include "solver/cellSolver.hpp"
#include "utilities/mathUtilities.hpp"

namespace ablate::eos::radiationProperties {
/** A radiation soot absorption model which computes the absorptivity based on the presence soot. */
class Rosseland : public RadiationModel {
   private:
    struct FunctionContext {
        PetscInt densityYiH2OOffset;
        PetscInt densityYiCO2Offset;
        PetscInt densityYiCOOffset;
        PetscInt densityYiCH4Offset;
        const ThermodynamicFunction temperatureFunction;
        const ThermodynamicFunction densityFunction;
    };

    double kapparef = 1;  //!< Reference absorptivity
    double Tsurf = 300.;  //!< Reference temperature in Kelvin

    /** The Rosseland model uses a fit approximation of the absorptivity. This depends on the presence of four species which are present in combustion and shown below. */
    double kappaH2O = 0;
    double kappaCO2 = 0;
    double kappaCH4 = 0;
    double kappaCO = 0;
    double pCO2, pH2O, pCH4, pCO;

    /** These are the constants stored for the model. */
    std::array<double, 7> H2O_coeff[] = {{0.22317e1, -.15829e1, .1329601e1, -.50707, .93334e-1, -0.83108e-2, 0.28834e-3}};       //
    std::array<double, 7> CO2_coeff[] = {{0.38041E1, -0.27808E1, 0.11672E1, -0.284910E0, 0.38163E-1, -0.26292E-2, 0.73662E-4}};  //!< polynomial coefficients for H2O
    std::array<double, 5> CH4_coeff[] = {{6.6334E0, -3.5686E-3, 1.6682E-8, 2.5611E-10, -2.6558E-14}};                            //!< polynomial coefficients for CH4
    std::array<double, 5> CO_1_coeff[] = {{4.7869E0, -6.953E-2, 2.95775E-4, -4.25732E-7, 2.02894E-10}};                          //!< polynomial coefficients for CO with T <= 750 K
    std::array<double, 5> CO_2_coeff[] = {{1.0091E1, -1.183E-2, 4.7753E-6, -5.87209E-10, -2.5334E-14}};                          //!< polynomial coefficients for CO with T > 750 K
    double MWC = 1.2010700e+01;
    double MWO = 1.599940e+01;
    double MWH = 1.007940e+00;
    double UGC = 8314.4;  //!< Universal Gas Constant J/(K kmol)
    double MWCO = MWC + MWO;
    double MWCO2 = MWC + 2. * MWO;
    double MWCH4 = MWC + 4. * MWH;
    double MWH2O = 2. * MWH + MWO;

    /**
     * Eos is needed to compute field values
     */
    const std::shared_ptr<eos::EOS> eos;

    /**
     * private static function for evaluating constant properties without temperature
     * @param conserved
     * @param property
     * @param ctx
     */
    static PetscErrorCode RosselandFunction(const PetscReal conserved[], PetscReal* property, void* ctx);

    /**
     * private static function for evaluating constant properties without temperature
     * @param conserved
     * @param property
     * @param ctx
     */
    static PetscErrorCode RosselandTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);

   public:
    explicit Rosseland(std::shared_ptr<eos::EOS> eosIn);
    explicit Rosseland(const Rosseland&) = delete;
    void operator=(const Rosseland&) = delete;

    /**
     * Single function to produce radiation properties function for any property based upon the available fields
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] ThermodynamicFunction GetRadiationPropertiesFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const override;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields and temperature
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] ThermodynamicTemperatureFunction GetRadiationPropertiesTemperatureFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const override;

    PetscInt GetFieldComponentOffset(const std::string& str, const domain::Field& field) const;
};

}  // namespace ablate::eos::radiationProperties

#endif  // ABLATELIBRARY_RADIATIONPROPERTIESROSSELAND_H