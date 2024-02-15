#ifndef ABLATELIBRARY_ZERORK_HPP
#define ABLATELIBRARY_ZERORK_HPP

#include <filesystem>
#include <map>
#include <memory>
#include "TChem_KineticModelData.hpp"
#include "chemistryModel.hpp"
#include "eos.hpp"
#include "eos/tChem/pressure.hpp"
#include "eos/tChem/sensibleEnthalpy.hpp"
#include "eos/tChem/sensibleInternalEnergy.hpp"
#include "eos/tChem/sourceCalculator.hpp"
#include "eos/tChem2/sourceCalculator2.hpp"
#include "eos/tChem/speedOfSound.hpp"
#include "eos/tChem/temperature.hpp"
#include "monitors/logs/log.hpp"
#include "parameters/parameters.hpp"
#include "utilities/intErrorChecker.hpp"
#include "utilities/vectorUtilities.hpp"
#include "zerork_cfd_plugin.h"

namespace ablate::eos {


class zerork : public ChemistryModel {
   protected:

    //create the plugin:
    zerork_handle zrm_handle;

    //! hold a copy of the constrains that can be used for single or batch source calculation
//    tChem::SourceCalculator::ChemistryConstraints constraints;

    const std::filesystem::path reactionFile;

    const std::filesystem::path thermoFile;

    const std::vector<std::string> species;
    struct constraints {
        PetscReal gamma;
        PetscReal rGas;
        PetscInt numberSpecies;
    };


    //! an optional log file for tchem echo redirection
    std::shared_ptr<ablate::monitors::logs::Log> log;

    /**
     * The kinetic model data
     */
//    tChemLib::KineticModelData kineticsModel;



   public:


    /**
     * The tChem EOS can utilize either a mechanical & thermo file using the Chemkin file format for a modern yaml file.
     * @param mechFile
     * @param optionalThermoFile
     */
    explicit zerork(const std::string& eosName, std::filesystem::path reactionFile, std::filesystem::path thermoFile, const std::shared_ptr<ablate::monitors::logs::Log>& = {},
                       const std::shared_ptr<ablate::parameters::Parameters>& options = {});



    /**
     * Returns all elements tracked in this mechanism and their molecular mass
     * @return
     */
    [[nodiscard]] virtual std::map<std::string, double> GetElementInformation() const = 0;

    /**
     * no. of atoms of each element in each species
     * @return
     */
    [[nodiscard]] virtual std::map<std::string, std::map<std::string, int>> GetSpeciesElementalInformation() const = 0;

    /**
     * the MW of each species
     * @return
     */
    [[nodiscard]] virtual std::map<std::string, double> GetSpeciesMolecularMass() const = 0;

    /**
     * Print the details of this eos
     * @param stream
     */
    void View(std::ostream& stream) const override;

    /**
     * return reference to kinetic data for other users
     */
//    tChemLib::KineticModelData& GetKineticModelData() { return kineticsModel; }

    /**
     * Get the  reference enthalpy per species
     */
//    real_type_1d_view GetEnthalpyOfFormation() { return enthalpyReferenceDevice; };

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
     * only allow modern input files
     */
    static const inline std::array<std::string, 2> validFileExtensions = {".yaml", ".yml"};

    struct FunctionContext {
        // memory access locations for fields
        PetscInt dim;
        PetscInt eulerOffset;
        PetscInt densityYiOffset;

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

    /**
     * a temperature thermodynamic function specific to TChem that takes yi from the arguments instead of conserved
     */
    struct ThermodynamicTemperatureMassFractionFunction {
        //! function to be called
        PetscErrorCode (*function)(const PetscReal conserved[], const PetscReal yi[], PetscReal T, PetscReal* property, void* ctx) = nullptr;
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
    [[nodiscard]] ThermodynamicTemperatureMassFractionFunction GetThermodynamicTemperatureMassFractionFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const;

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
         {PressureFunction, PressureTemperatureFunction, ablate::eos::tChem::Temperature::getWorkSpaceSize}} /**note size of temperature because it has a larger scratch space */,
        {ThermodynamicProperty::Temperature, {TemperatureFunction, TemperatureTemperatureFunction, ablate::eos::tChem::Temperature::getWorkSpaceSize}},
        {ThermodynamicProperty::InternalSensibleEnergy, {InternalSensibleEnergyFunction, InternalSensibleEnergyTemperatureFunction, ablate::eos::tChem::SensibleInternalEnergy::getWorkSpaceSize}},
        {ThermodynamicProperty::SensibleEnthalpy, {SensibleEnthalpyFunction, SensibleEnthalpyTemperatureFunction, ablate::eos::tChem::SensibleEnthalpy::getWorkSpaceSize}},
        {ThermodynamicProperty::SpecificHeatConstantVolume, {SpecificHeatConstantVolumeFunction, SpecificHeatConstantVolumeTemperatureFunction, [](auto nSpec) { return nSpec; }}},
        {ThermodynamicProperty::SpecificHeatConstantPressure,
         {SpecificHeatConstantPressureFunction,
          SpecificHeatConstantPressureTemperatureFunction,
          ablate::eos::tChem::Temperature::getWorkSpaceSize}} /**note size of temperature because it has a larger scratch space */,
        {ThermodynamicProperty::SpeedOfSound, {SpeedOfSoundFunction, SpeedOfSoundTemperatureFunction, ablate::eos::tChem::SpeedOfSound::getWorkSpaceSize}},
        {ThermodynamicProperty::SpeciesSensibleEnthalpy,
         {SpeciesSensibleEnthalpyFunction,
          SpeciesSensibleEnthalpyTemperatureFunction,
          ablate::eos::tChem::Temperature::getWorkSpaceSize}} /**note size of temperature because it has a larger scratch space */
    };
};

}  // namespace ablate::eos
#endif  // ABLATELIBRARY_ERORK_HPP
