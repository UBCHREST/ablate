#ifndef ABLATELIBRARY_TCHEMSOOT_HPP
#define ABLATELIBRARY_TCHEMSOOT_HPP

#include <filesystem>
#include <map>
#include <memory>
#include "TChem_KineticModelData.hpp"
#include "chemistryModel.hpp"
#include "eos.hpp"
#include "eos/tChemSoot/pressure.hpp"
#include "eos/tChemSoot/sensibleEnthalpy.hpp"
#include "eos/tChemSoot/sensibleInternalEnergy.hpp"
#include "eos/tChemSoot/speedOfSound.hpp"
#include "eos/tChemSoot/stateVectorSoot.hpp"
#include "eos/tChemSoot/temperature.hpp"
#include "monitors/logs/log.hpp"
#include "tChem.hpp"
#include "utilities/intErrorChecker.hpp"
#include <set>

namespace ablate::eos {

namespace tChemLib = TChem;

class TChemSoot : public TChemBase, public std::enable_shared_from_this<ablate::eos::TChemSoot> {
   public:
    /**
     * The name of the solid species
     */
    static inline std::string CSolidName = "C_solid";

    /**
     * The name of the Soot Number Density progress variable
     */
    static inline std::string SootNumberDensityName = "SootNumberDensity_Mass";

   private:
    /**
     * keep the progress variables names as
     */
    std::vector<std::string> progressVariables = {SootNumberDensityName};

   public:
    /**
     * The tChem EOS can utilize either a mechanical & thermo file using the Chemkin file format for a modern yaml file.
     * @param mechFile
     * @param optionalThermoFile
     */
    explicit TChemSoot(std::filesystem::path mechanismFile, std::filesystem::path thermoFile = {}, std::shared_ptr<ablate::monitors::logs::Log> = {},
                       const std::shared_ptr<ablate::parameters::Parameters>& options = {});

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] ThermodynamicFunction GetThermodynamicFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields and temperature
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] ThermodynamicTemperatureFunction GetThermodynamicTemperatureFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override;

    /**
     * Single function to produce fieldFunction function for any two properties, velocity, and species mass fractions.  These calls can be slower and should be used for init/output only
     * @param field
     * @param property1
     * @param property2
     */
    [[nodiscard]] EOSFunction GetFieldFunctionFunction(const std::string& field, ThermodynamicProperty property1, ThermodynamicProperty property2,
                                                       std::vector<std::string> otherProperties) const override;

    /**
     * TThe tChemSoot also need to track the soot number density
     * @return
     */
    [[nodiscard]] const std::vector<std::string>& GetProgressVariables() const override { return progressVariables; }

    /**
     * Returns all elements tracked in this mechanism and their molecular mass
     * @return
     */
    [[nodiscard]] std::map<std::string, double> GetElementInformation() const override;

    /**
     * no. of atoms of each element in each species
     * @return
     */
    [[nodiscard]] std::map<std::string, std::map<std::string, int>> GetSpeciesElementalInformation() const override;

    /**
     * the MW of each species
     * @return
     */
    [[nodiscard]] std::map<std::string, double> GetSpeciesMolecularMass() const override;

    /**
     * Print the details of this eos
     * @param stream
     */
    void View(std::ostream& stream) const override;

    /**
     * Function to create the batch source specific to the provided cell range
     * @param cellRange
     * @return
     */
    std::shared_ptr<SourceCalculator> CreateSourceCalculator(const std::vector<domain::Field>& fields, const solver::Range& cellRange) override;

    /**
     * helper function to build the function context needed regardless of function type.  This is specialized for soot
     * @tparam Function
     * @tparam Type
     * @param fields
     * @return
     */
    [[nodiscard]] std::shared_ptr<FunctionContext> BuildFunctionContext(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field>& fields) const;

    /** @name Direct Thermodynamic Properties Functions
     * These functions are used to compute the direct thermodynamic properties (without temperature).  They are not called directly but a pointer to them is returned
     * @param conserved
     * @param property
     * @param ctx
     * @return
     * @{
     */
    static PetscErrorCode DensityFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode TemperatureFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode PressureFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode InternalSensibleEnergyFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SensibleEnthalpyFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantVolumeFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantPressureFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpeedOfSoundFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpeciesSensibleEnthalpyFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    /** @} */

    /** @name Temperature Based Thermodynamic Properties Functions
     * These functions are used to compute the thermodynamic properties when temperature is known.  They are not called directly but a pointer to them is returned and may be faster than the direct
     * calls.
     * @param conserved
     * @param property
     * @param ctx
     * @return
     * @{
     */
    static PetscErrorCode DensityTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode TemperatureTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode PressureTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode InternalSensibleEnergyTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SensibleEnthalpyTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantVolumeTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantPressureTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpeedOfSoundTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpeciesSensibleEnthalpyTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    /** @} */

    /**
     * template function to call base tChem function
     */

    /**
     * Store a map of functions functions for quick lookup
     */
    using ThermodynamicStaticFunction = PetscErrorCode (*)(const PetscReal conserved[], PetscReal* property, void* ctx);
    using ThermodynamicTemperatureStaticFunction = PetscErrorCode (*)(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);
    std::map<ThermodynamicProperty, std::tuple<ThermodynamicStaticFunction, ThermodynamicTemperatureStaticFunction, std::function<ordinal_type(ordinal_type)>>> thermodynamicFunctions = {
        {ThermodynamicProperty::Density, {DensityFunction, DensityTemperatureFunction, [](auto) { return 0; }}},
        {ThermodynamicProperty::Pressure,
         {PressureFunction, PressureTemperatureFunction, ablate::eos::tChemSoot::Temperature::getWorkSpaceSize}} /**note size of temperature because it has a larger scratch space */,
        {ThermodynamicProperty::Temperature, {TemperatureFunction, TemperatureTemperatureFunction, ablate::eos::tChemSoot::Temperature::getWorkSpaceSize}},
        {ThermodynamicProperty::InternalSensibleEnergy, {InternalSensibleEnergyFunction, InternalSensibleEnergyTemperatureFunction, ablate::eos::tChemSoot::SensibleInternalEnergy::getWorkSpaceSize}},
        {ThermodynamicProperty::SensibleEnthalpy, {SensibleEnthalpyFunction, SensibleEnthalpyTemperatureFunction, ablate::eos::tChemSoot::SensibleEnthalpy::getWorkSpaceSize}},
        {ThermodynamicProperty::SpecificHeatConstantVolume, {SpecificHeatConstantVolumeFunction, SpecificHeatConstantVolumeTemperatureFunction, [](auto nSpec) { return nSpec; }}},
        {ThermodynamicProperty::SpecificHeatConstantPressure,
         {SpecificHeatConstantPressureFunction,
          SpecificHeatConstantPressureTemperatureFunction,
          ablate::eos::tChemSoot::Temperature::getWorkSpaceSize}} /**note size of temperature because it has a larger scratch space */,
        {ThermodynamicProperty::SpeedOfSound, {SpeedOfSoundFunction, SpeedOfSoundTemperatureFunction, ablate::eos::tChemSoot::SpeedOfSound::getWorkSpaceSize}},
        {ThermodynamicProperty::SpeciesSensibleEnthalpy,
         {SpeciesSensibleEnthalpyFunction,
          SpeciesSensibleEnthalpyTemperatureFunction,
          ablate::eos::tChemSoot::Temperature::getWorkSpaceSize}} /**note size of temperature because it has a larger scratch space */
    };

    /**
     * Store a list of properties that are sized by species, everything is assumed to be size one
     */
     const std::set<ThermodynamicProperty> speciesSizedProperties = {ThermodynamicProperty::SpeciesSensibleEnthalpy};

    /**
     * Fill and Normalize the density species mass fractions
     * @param numSpec
     * @param yi
     */
    static void FillWorkingVectorFromDensityMassFractions(double density, double temperature, const double* densityYi, const tChemSoot::StateVectorSoot<real_type_1d_view_host>& stateVector);

    // New stuff
   private:
    // Graphite date
    inline static const std::vector<real_type> CS_Nasa7TLow = {-3.108720720e-01, 4.403536860e-03, 1.903941180e-06, -6.385469660e-09, 2.989642480e-12, -1.086507940e+02, 1.113829530e+00};
    inline static const std::vector<real_type> CS_Nasa7THigh = {1.455718290e+00, 1.717022160e-03, -6.975627860e-07, 1.352770320e-10, -9.675906520e-15, -6.951388140e+02, -8.525830330e+00};

   public:
    inline static real_type CarbonEnthalpy_R_T(const real_type& Temp) {
        if ((Temp < 200.)) {
            double t200 = CS_Nasa7TLow.at(5) / 200. + CS_Nasa7TLow.at(0) +
                          200. * (CS_Nasa7TLow.at(1) / 2. + 200. * (CS_Nasa7TLow.at(2) / 3. + 200. * (CS_Nasa7TLow.at(3) / 4. + 200. * CS_Nasa7TLow.at(4) / 5.)));
            double t300 = CS_Nasa7TLow.at(5) / 300. + CS_Nasa7TLow.at(0) +
                          300. * (CS_Nasa7TLow.at(1) / 2. + 300. * (CS_Nasa7TLow.at(2) / 3. + 300. * (CS_Nasa7TLow.at(3) / 4. + 300. * CS_Nasa7TLow.at(4) / 5.)));
            return t200 - (t300 - t200) / 100. * (200. - Temp);
        } else if (Temp <= 1000.)
            return CS_Nasa7TLow.at(5) / Temp + CS_Nasa7TLow.at(0) +
                   Temp * (CS_Nasa7TLow.at(1) / 2. + Temp * (CS_Nasa7TLow.at(2) / 3. + Temp * (CS_Nasa7TLow.at(3) / 4. + Temp * CS_Nasa7TLow.at(4) / 5.)));
        else if (Temp <= 5000.)
            return CS_Nasa7THigh.at(5) / Temp + CS_Nasa7THigh.at(0) +
                   Temp * (CS_Nasa7THigh.at(1) / 2. + Temp * (CS_Nasa7THigh.at(2) / 3. + Temp * (CS_Nasa7THigh.at(3) / 4. + Temp * CS_Nasa7THigh.at(4) / 5.)));
        else {
            double t5000 = CS_Nasa7THigh.at(5) / 5000. + CS_Nasa7THigh.at(0) +
                           5000. * (CS_Nasa7THigh.at(1) / 2. + 5000. * (CS_Nasa7THigh.at(2) / 3. + 5000. * (CS_Nasa7THigh.at(3) / 4. + 5000. * CS_Nasa7THigh.at(4) / 5.)));
            double t4900 = CS_Nasa7THigh.at(5) / 4900. + CS_Nasa7THigh.at(0) +
                           4900. * (CS_Nasa7THigh.at(1) / 2. + 4900. * (CS_Nasa7THigh.at(2) / 3. + 4900. * (CS_Nasa7THigh.at(3) / 4. + 4900. * CS_Nasa7THigh.at(4) / 5.)));
            return t5000 + (t5000 - t4900) / 100. * (Temp - 5000.);
        }
    }

    inline static double CarbonCp_R(const double& Temp) {
        if (Temp < 200) {
            double t200 = CS_Nasa7TLow.at(0) + 200. * (CS_Nasa7TLow.at(1) + 200. * (CS_Nasa7TLow.at(2) + 200. * (CS_Nasa7TLow.at(3) + 200. * CS_Nasa7TLow.at(4))));
            double t300 = CS_Nasa7TLow.at(0) + 300. * (CS_Nasa7TLow.at(1) + 300. * (CS_Nasa7TLow.at(2) + 300. * (CS_Nasa7TLow.at(3) + 300. * CS_Nasa7TLow.at(4))));
            return t200 - (t300 - t200) / 100. * (200 - Temp);
        } else if (Temp <= 1000)
            return CS_Nasa7TLow.at(0) + Temp * (CS_Nasa7TLow.at(1) + Temp * (CS_Nasa7TLow.at(2) + Temp * (CS_Nasa7TLow.at(3) + Temp * CS_Nasa7TLow.at(4))));
        else if (Temp <= 5000)
            return CS_Nasa7THigh.at(0) + Temp * (CS_Nasa7THigh.at(1) + Temp * (CS_Nasa7THigh.at(2) + Temp * (CS_Nasa7THigh.at(3) + Temp * CS_Nasa7THigh.at(4))));
        else {
            double t5000 = CS_Nasa7THigh.at(0) + 5000. * (CS_Nasa7THigh.at(1) + 5000. * (CS_Nasa7THigh.at(2) + 5000. * (CS_Nasa7THigh.at(3) + 5000. * CS_Nasa7THigh.at(4))));
            double t4900 = CS_Nasa7THigh.at(0) + 4900. * (CS_Nasa7THigh.at(1) + 4900. * (CS_Nasa7THigh.at(2) + 4900. * (CS_Nasa7THigh.at(3) + 4900. * CS_Nasa7THigh.at(4))));
            return t5000 + (t5000 - t4900) / 100. * (Temp - 5000.);
        }
    }
};

}  // namespace ablate::eos
#endif  // ABLATELIBRARY_TCHEM_HPP
