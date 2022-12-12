#ifndef ABLATELIBRARY_PERFECTGAS_HPP
#define ABLATELIBRARY_PERFECTGAS_HPP
#include <map>
#include <memory>
#include "eos.hpp"
#include "parameters/parameters.hpp"
#include "utilities/vectorUtilities.hpp"

namespace ablate::eos {

class PerfectGas : public EOS {
   private:
    // the perfect gas does not allow species
    const std::vector<std::string> species;
    struct Parameters {
        PetscReal gamma;
        PetscReal rGas;
        PetscInt numberSpecies;
    };
    Parameters parameters{};

    struct FunctionContext {
        PetscInt dim;
        PetscInt eulerOffset;
        Parameters parameters;
    };

    /** @name Direct Thermodynamic Properties Functions
     * These functions are used to compute the direct thermodynamic properties (without temperature).  They are not called directly but a pointer to them is returned
     * @param conserved
     * @param property
     * @param ctx
     * @return
     * @{
     */
    static PetscErrorCode DensityFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode PressureFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode TemperatureFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
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
    static PetscErrorCode PressureTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode TemperatureTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
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
    explicit PerfectGas(const std::shared_ptr<ablate::parameters::Parameters>&, std::vector<std::string> species = {});
    void View(std::ostream& stream) const override;
    /**
     * Get constant specific heat ratio for a perfect gas.
     * @return
     */
    [[nodiscard]] PetscReal GetSpecificHeatRatio() const { return parameters.gamma; }

    /**
     * Get constant gas constant for a perfect gas
     * @return
     */
    [[nodiscard]] PetscReal GetGasConstant() const { return parameters.rGas; }

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
    [[nodiscard]] FieldFunction GetFieldFunctionFunction(const std::string& field, ThermodynamicProperty property1, ThermodynamicProperty property2) const override;

    /**
     * returns the species supported by this EOS
     * @return
     */
    [[nodiscard]] const std::vector<std::string>& GetSpeciesVariables() const override { return species; }

    /**
     * Returns a vector of all extra variables required to utilize the equation of state
     * @return
     */
    [[nodiscard]] virtual const std::vector<std::string>& GetProgressVariables() const override { return ablate::utilities::VectorUtilities::Empty<std::string>; }
};

}  // namespace ablate::eos
#endif  // ABLATELIBRARY_PERFECTGAS_HPP
