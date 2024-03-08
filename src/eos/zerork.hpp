#ifndef ABLATELIBRARY_ZERORKEOS_HPP
#define ABLATELIBRARY_ZERORKEOS_HPP

#include <filesystem>
#include <map>
#include <memory>
#include "chemistryModel.hpp"
#include "eos.hpp"
#include "eos/zerork/sourceCalculatorZeroRK.hpp"
#include "monitors/logs/log.hpp"
#include "parameters/parameters.hpp"
#include "utilities/intErrorChecker.hpp"
#include "utilities/vectorUtilities.hpp"
#include "zerork_cfd_plugin.h"
#include "zerork/mechanism.h"
#include "zerork/utilities.h"

namespace ablate::eos {

class zerorkEOS : public ChemistryModel, public std::enable_shared_from_this<ablate::eos::zerorkEOS>{
   protected:

    //! hold a copy of the constrains that can be used for single or batch source calculation
    zerorkeos::SourceCalculator::ChemistryConstraints constraints;

    std::vector<std::string> species;

    int nSpc;

    struct constraints {
        PetscInt numberSpecies;
    };

    //kinetic and elemental data parsing file
    const char* cklogfilename = "mech.cklog";

    std::vector<double> stateVector;

   public:

    /**
     * Zerork only takes in chemkin formated reaction files and thermodynamic files.
     * @param mechFile
     * @param optionalThermoFile
     */
    explicit zerorkEOS(std::filesystem::path reactionFile, std::filesystem::path thermoFile,
                       const std::shared_ptr<ablate::parameters::Parameters>& options = {});

    std::shared_ptr<zerork::mechanism> mech;

    const std::filesystem::path reactionFile;

    const std::filesystem::path thermoFile;
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
     * Species supported by this EOS
     * species model functions
     * @return
     */
    [[nodiscard]] const std::vector<std::string>& GetSpeciesVariables() const override { return species; }

    /**
     * Returns a vector of all extra variables required to utilize the equation of state
     * @return
     */
    [[nodiscard]] virtual const std::vector<std::string>& GetProgressVariables() const override { return ablate::utilities::VectorUtilities::Empty<std::string>; }

    /**
     * Helper function to get the specific EnthalpyOfFormation
     * @param species
     * @return
     */


   protected:
    /**
     * only allow chemkin input files
     */

    static const inline std::array<std::string, 2> validThermoFileExtensions = {".dat",".log"};
//    static const std::string validChemkinFileExtensions = ".inp";


    struct FunctionContext {
        // memory access locations for fields
        PetscInt dim;
        PetscInt eulerOffset;
        PetscInt densityYiOffset;

        std::shared_ptr<zerork::mechanism> mech;
        PetscInt nSpc;
    };

   public:
    // Private static helper functions
    inline const static double TREF = 298.15;

    struct ThermodynamicMassFractionFunction {
        //! function to be called
        PetscErrorCode (*function)(const PetscReal conserved[], const PetscReal yi[], PetscReal* property, void* ctx) = nullptr;
        //! optional context to pass into the function
        std::shared_ptr<void> context = nullptr;
        //! the property size being set
        PetscInt propertySize = 1;
    };


   public:

    [[nodiscard]] ThermodynamicFunction GetThermodynamicFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields and temperature
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] ThermodynamicTemperatureFunction GetThermodynamicTemperatureFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields and yi
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] ThermodynamicMassFractionFunction GetThermodynamicMassFractionFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields, temperature and yi
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] ThermodynamicTemperatureMassFractionFunction GetThermodynamicTemperatureMassFractionFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override;

    /**
     * Single function to produce fieldFunction function for any two properties, velocity, and species mass fractions.  These calls can be slower and should be used for init/output only
     * @param field
     * @param property1
     * @param property2
     */
    [[nodiscard]] EOSFunction GetFieldFunctionFunction(const std::string& field, ThermodynamicProperty property1, ThermodynamicProperty property2,
                                                       std::vector<std::string> otherProperties) const override;


    /**
     * Function to create the batch source specific to the provided cell range
     * @param fields
     * @param cellRange
     * @return
     */
    std::shared_ptr<SourceCalculator> CreateSourceCalculator(const std::vector<domain::Field>& fields, const ablate::domain::Range& cellRange) override;

   protected:
    /**
     * helper function to build the function context needed regardless of function type
     * @tparam Function
     * @tparam Type
     * @param fields
     * @param optional argument to check for yi
     * @return
     */
    [[nodiscard]] std::shared_ptr<FunctionContext> BuildFunctionContext(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field>& fields, bool checkDensityYi = true) const;

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

    /** @name Direct Thermodynamic Properties Functions
     * These functions are used to compute the direct thermodynamic properties (without temperature).  They are not called directly but a pointer to them is returned
     * @param conserved
     * @param property
     * @param ctx
     * @return
     * @{
     */
    static PetscErrorCode DensityMassFractionFunction(const PetscReal conserved[], const PetscReal yi[], PetscReal* property, void* ctx);
    static PetscErrorCode TemperatureMassFractionFunction(const PetscReal conserved[], const PetscReal yi[], PetscReal* property, void* ctx);
    static PetscErrorCode PressureMassFractionFunction(const PetscReal conserved[], const PetscReal yi[], PetscReal* property, void* ctx);
    static PetscErrorCode InternalSensibleEnergyMassFractionFunction(const PetscReal conserved[], const PetscReal yi[], PetscReal* property, void* ctx);
    static PetscErrorCode SensibleEnthalpyMassFractionFunction(const PetscReal conserved[], const PetscReal yi[], PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantVolumeMassFractionFunction(const PetscReal conserved[], const PetscReal yi[], PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantPressureMassFractionFunction(const PetscReal conserved[], const PetscReal yi[], PetscReal* property, void* ctx);
    static PetscErrorCode SpeedOfSoundMassFractionFunction(const PetscReal conserved[], const PetscReal yi[], PetscReal* property, void* ctx);
    static PetscErrorCode SpeciesSensibleEnthalpyMassFractionFunction(const PetscReal conserved[], const PetscReal yi[], PetscReal* property, void* ctx);
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
    static PetscErrorCode DensityTemperatureMassFractionFunction(const PetscReal conserved[], const PetscReal yi[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode TemperatureTemperatureMassFractionFunction(const PetscReal conserved[], const PetscReal yi[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode PressureTemperatureMassFractionFunction(const PetscReal conserved[], const PetscReal yi[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode InternalSensibleEnergyTemperatureMassFractionFunction(const PetscReal conserved[], const PetscReal yi[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SensibleEnthalpyTemperatureMassFractionFunction(const PetscReal conserved[], const PetscReal yi[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantVolumeTemperatureMassFractionFunction(const PetscReal conserved[], const PetscReal yi[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantPressureTemperatureMassFractionFunction(const PetscReal conserved[], const PetscReal yi[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpeedOfSoundTemperatureMassFractionFunction(const PetscReal conserved[], const PetscReal yi[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpeciesSensibleEnthalpyTemperatureMassFractionFunction(const PetscReal conserved[], const PetscReal yi[], PetscReal T, PetscReal* property, void* ctx);
    /** @} */

    /**
     * Store a map of functions functions for quick lookup
     */
    using ThermodynamicStaticFunction = PetscErrorCode (*)(const PetscReal conserved[], PetscReal* property, void* ctx);
    using ThermodynamicTemperatureStaticFunction = PetscErrorCode (*)(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);
    std::map<ThermodynamicProperty, std::tuple<ThermodynamicStaticFunction, ThermodynamicTemperatureStaticFunction>> thermodynamicFunctions = {
        {ThermodynamicProperty::Density, {DensityFunction, DensityTemperatureFunction}},
        {ThermodynamicProperty::Pressure,
         {PressureFunction, PressureTemperatureFunction}},
        {ThermodynamicProperty::Temperature, {TemperatureFunction, TemperatureTemperatureFunction}},
        {ThermodynamicProperty::InternalSensibleEnergy, {InternalSensibleEnergyFunction, InternalSensibleEnergyTemperatureFunction}},
        {ThermodynamicProperty::SensibleEnthalpy, {SensibleEnthalpyFunction, SensibleEnthalpyTemperatureFunction}},
        {ThermodynamicProperty::SpecificHeatConstantVolume, {SpecificHeatConstantVolumeFunction, SpecificHeatConstantVolumeTemperatureFunction}},
        {ThermodynamicProperty::SpecificHeatConstantPressure,
         {SpecificHeatConstantPressureFunction,
          SpecificHeatConstantPressureTemperatureFunction}},
        {ThermodynamicProperty::SpeedOfSound, {SpeedOfSoundFunction, SpeedOfSoundTemperatureFunction}},
        {ThermodynamicProperty::SpeciesSensibleEnthalpy,
         {SpeciesSensibleEnthalpyFunction,
          SpeciesSensibleEnthalpyTemperatureFunction}} /**note size of temperature because it has a larger scratch space */
    };


    using ThermodynamicStaticMassFractionFunction = PetscErrorCode (*)(const PetscReal conserved[], const PetscReal yi[], PetscReal* property, void* ctx);
    using ThermodynamicTemperatureStaticMassFractionFunction = PetscErrorCode (*)(const PetscReal conserved[], const PetscReal yi[], PetscReal temperature, PetscReal* property, void* ctx);
    std::map<ThermodynamicProperty, std::tuple<ThermodynamicStaticMassFractionFunction, ThermodynamicTemperatureStaticMassFractionFunction>>
        thermodynamicMassFractionFunctions = {
            {ThermodynamicProperty::Density, {DensityMassFractionFunction, DensityTemperatureMassFractionFunction}},
            {ThermodynamicProperty::Pressure,
             {PressureMassFractionFunction,
              PressureTemperatureMassFractionFunction}} /**note size of temperature because it has a larger scratch space */,
            {ThermodynamicProperty::Temperature, {TemperatureMassFractionFunction, TemperatureTemperatureMassFractionFunction}},
            {ThermodynamicProperty::InternalSensibleEnergy,
             {InternalSensibleEnergyMassFractionFunction, InternalSensibleEnergyTemperatureMassFractionFunction}},
            {ThermodynamicProperty::SensibleEnthalpy, {SensibleEnthalpyMassFractionFunction, SensibleEnthalpyTemperatureMassFractionFunction}},
            {ThermodynamicProperty::SpecificHeatConstantVolume,
             {SpecificHeatConstantVolumeMassFractionFunction, SpecificHeatConstantVolumeTemperatureMassFractionFunction}},
            {ThermodynamicProperty::SpecificHeatConstantPressure,
             {SpecificHeatConstantPressureMassFractionFunction,
              SpecificHeatConstantPressureTemperatureMassFractionFunction}} /**note size of temperature because it has a larger scratch space */,
            {ThermodynamicProperty::SpeedOfSound, {SpeedOfSoundMassFractionFunction, SpeedOfSoundTemperatureMassFractionFunction}},
            {ThermodynamicProperty::SpeciesSensibleEnthalpy,
             {SpeciesSensibleEnthalpyMassFractionFunction,
              SpeciesSensibleEnthalpyTemperatureMassFractionFunction}} /**note size of temperature because it has a larger scratch space */
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
    static void FillreactorMassFracVectorFromDensityMassFractions(int nSpc,double density, const double* densityYi, std::vector<double>& reactorYi);

    /**
     * Fill the working vector from yi
     * @param numSpec
     * @param yi
     */
    static void FillreactorMassFracVectorFromMassFractions(int nSpc,const double* Yi, std::vector<double>& reactorYi);
};

}  // namespace ablate::eos
#endif  // ABLATELIBRARY_ERORK_HPP
