#ifndef ABLATECLIENTTEMPLATE_TCHEM_HPP
#define ABLATECLIENTTEMPLATE_TCHEM_HPP

#include <filesystem>
#include <map>
#include "eos.hpp"
#include "utilities/intErrorChecker.hpp"

namespace ablate::eos {

#define TCCHKERRQ(ierr)                                                                                    \
    do {                                                                                                   \
        if (ierr) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in TChem library, return code %d", ierr); \
    } while (0)

class TChem : public EOS {
   private:
    // this is bad practice but only one instance of the TCHEM library can be inited at once, so keep track of the number of classes using the library and prevent multiple uses
    inline static bool libUsed = false;

    // hold an error checker for the tchem outside library
    const utilities::IntErrorChecker errorChecker;

    // path to the input files
    std::filesystem::path mechFile;
    std::filesystem::path thermoFile;

    // prestore all species
    std::vector<std::string> species;
    int numberSpecies;

    // store a tcWorkingVector
    mutable std::vector<double> tempYiWorkingVector;
    mutable std::vector<double> sourceWorkingVector;

    // precompute the speciesHeatOfFormation taken at TREF
    std::vector<double> speciesHeatOfFormation;

    // write/reproduce the periodic table
    static const char* periodicTable;
    inline static const char* periodicTableFileName = "periodictable.dat";

    struct FunctionContext {
        PetscInt dim;
        PetscInt eulerOffset;
        PetscInt densityYiOffset;
        const TChem* tChem;
    };

    /**
     * The tempYiWorkingArray is expected to be filled with correct species yi.  The 0 location is set in this function.
     * @param numSpec
     * @param tempYiWorkingArray
     * @param internalEnergyRef
     * @param mwMix
     * @param T
     * @return
     */
    static PetscErrorCode ComputeTemperatureInternal(int numSpec, double* tempYiWorkingArray, PetscReal internalEnergyRef, double mwMix, double& T);

    /**
     * the tempYiWorkingArray array is expected to be filled
     * @param numSpec
     * @param tempYiWorkingArray
     * @param T
     * @param mwMix
     * @param internalEnergy
     * @return
     */
    static int ComputeSensibleInternalEnergyInternal(int numSpec, double* tempYiWorkingArray, double mwMix, double& internalEnergy);

    /**
     * Fill and normalize the species mass fractions
     * @param numSpec
     * @param yi
     */
    static void FillWorkingVectorFromMassFractions(int numSpec, double temperature, const double* yi, double* workingVector);

    /**
     * Fill and Normalize the density species mass fractions
     * @param numSpec
     * @param yi
     */
    static void FillWorkingVectorFromDensityMassFractions(int numSpec, double density, double temperature, const double* densityYi, double* workingVector);

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
     * Store a map of functions functions for quick lookup
     */
    using ThermodynamicStaticFunction = PetscErrorCode (*)(const PetscReal conserved[], PetscReal* property, void* ctx);
    using ThermodynamicTemperatureStaticFunction = PetscErrorCode (*)(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);
    std::map<ThermodynamicProperty, std::pair<ThermodynamicStaticFunction, ThermodynamicTemperatureStaticFunction>> thermodynamicFunctions = {
        {ThermodynamicProperty::Density, {DensityFunction, DensityTemperatureFunction}},
        {ThermodynamicProperty::Pressure, {PressureFunction, PressureTemperatureFunction}},
        {ThermodynamicProperty::Temperature, {TemperatureFunction, TemperatureTemperatureFunction}},
        {ThermodynamicProperty::InternalSensibleEnergy, {InternalSensibleEnergyFunction, InternalSensibleEnergyTemperatureFunction}},
        {ThermodynamicProperty::SensibleEnthalpy, {SensibleEnthalpyFunction, SensibleEnthalpyTemperatureFunction}},
        {ThermodynamicProperty::SpecificHeatConstantVolume, {SpecificHeatConstantVolumeFunction, SpecificHeatConstantVolumeTemperatureFunction}},
        {ThermodynamicProperty::SpecificHeatConstantPressure, {SpecificHeatConstantPressureFunction, SpecificHeatConstantPressureTemperatureFunction}},
        {ThermodynamicProperty::SpeedOfSound, {SpeedOfSoundFunction, SpeedOfSoundTemperatureFunction}},
        {ThermodynamicProperty::SpeciesSensibleEnthalpy, {SpeciesSensibleEnthalpyFunction, SpeciesSensibleEnthalpyTemperatureFunction}}};

   public:
    TChem(std::filesystem::path mechFile, std::filesystem::path thermoFile);
    ~TChem() override;

    // general functions
    void View(std::ostream& stream) const override;

    // species model functions
    const std::vector<std::string>& GetSpecies() const override;

    /**
     * Used by tChem Rxns
     * @param numSpec
     * @param tempYiWorkingArray
     * @param enthalpyOfFormation
     * @return
     */
    static int ComputeEnthalpyOfFormation(int numSpec, double* tempYiWorkingArray, double& enthalpyOfFormation);

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields
     * @param property
     * @param fields
     * @return
     */
    ThermodynamicFunction GetThermodynamicFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields and temperature
     * @param property
     * @param fields
     * @return
     */
    ThermodynamicTemperatureFunction GetThermodynamicTemperatureFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override;

    /**
     * Single function to produce fieldFunction function for any two properties, velocity, and species mass fractions.  These calls can be slower and should be used for init/output only
     * @param field
     * @param property1
     * @param property2
     */
    FieldFunction GetFieldFunctionFunction(const std::string& field, ThermodynamicProperty property1, ThermodynamicProperty property2) const override;

    // Private static helper functions
    inline const static double TREF = 298.15;
};

}  // namespace ablate::eos
#endif  // ABLATECLIENTTEMPLATE_TCHEM_HPP
