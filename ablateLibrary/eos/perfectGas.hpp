#ifndef ABLATELIBRARY_PERFECTGAS_HPP
#define ABLATELIBRARY_PERFECTGAS_HPP
#include <map>
#include <memory>
#include "eos.hpp"
#include "parameters/parameters.hpp"

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
    Parameters parameters;

    struct FunctionContext {
        PetscInt dim;
        PetscInt eulerOffset;
        Parameters parameters;
    };

    static PetscErrorCode PerfectGasDecodeState(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a,
                                                PetscReal* p, void* ctx);
    static PetscErrorCode PerfectGasComputeTemperature(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx);

    static PetscErrorCode PerfectGasComputeSpeciesSensibleEnthalpy(PetscReal T, PetscReal* hi, void* ctx);

    static PetscErrorCode PerfectGasComputeDensityFunctionFromTemperaturePressure(PetscReal T, PetscReal pressure, const PetscReal yi[], PetscReal* density, void* ctx);

    static PetscErrorCode PerfectGasComputeSensibleInternalEnergy(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleInternalEnergy, void* ctx);

    static PetscErrorCode PerfectGasComputeSensibleEnthalpy(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleInternalEnergy, void* ctx);

    static PetscErrorCode PerfectGasComputeSpecificHeatConstantPressure(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx);

    static PetscErrorCode PerfectGasComputeSpecificHeatConstantVolume(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx);

    static PetscErrorCode PressureFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode TemperatureFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode InternalSensibleEnergyFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SensibleEnthalpyFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantVolumeFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantPressureFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpeedOfSoundFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpeciesSensibleEnthalpyFunction(const PetscReal conserved[], PetscReal* property, void* ctx);

    static PetscErrorCode PressureTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode TemperatureTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode InternalSensibleEnergyTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SensibleEnthalpyTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantVolumeTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantPressureTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpeedOfSoundTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpeciesSensibleEnthalpyTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);

    /**
     * Store a map of functions functions for quick lookup
     */
    using ThermodynamicStaticFunction = PetscErrorCode (*)(const PetscReal conserved[], PetscReal* property, void* ctx);
    inline static std::map<ThermodynamicProperty, ThermodynamicStaticFunction> thermodynamicFunctions = {{ThermodynamicProperty::Pressure, PressureFunction},
                                                                                                         {ThermodynamicProperty::Temperature, TemperatureFunction},
                                                                                                         {ThermodynamicProperty::InternalSensibleEnergy, InternalSensibleEnergyFunction},
                                                                                                         {ThermodynamicProperty::SensibleEnthalpy, SensibleEnthalpyFunction},
                                                                                                         {ThermodynamicProperty::SpecificHeatConstantVolume, SpecificHeatConstantVolumeFunction},
                                                                                                         {ThermodynamicProperty::SpecificHeatConstantPressure, SpecificHeatConstantPressureFunction},
                                                                                                         {ThermodynamicProperty::SpeedOfSound, SpeedOfSoundFunction},
                                                                                                         {ThermodynamicProperty::SpeciesSensibleEnthalpy, SpeciesSensibleEnthalpyFunction}};

    /**
     * Store a map of temperature functions for quick lookup
     */
    using ThermodynamicTemperatureStaticFunction = PetscErrorCode (*)(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);
    inline static std::map<ThermodynamicProperty, ThermodynamicTemperatureStaticFunction> thermodynamicTemperatureFunctions = {
        {ThermodynamicProperty::Pressure, PressureTemperatureFunction},
        {ThermodynamicProperty::Temperature, TemperatureTemperatureFunction},
        {ThermodynamicProperty::InternalSensibleEnergy, InternalSensibleEnergyTemperatureFunction},
        {ThermodynamicProperty::SensibleEnthalpy, SensibleEnthalpyTemperatureFunction},
        {ThermodynamicProperty::SpecificHeatConstantVolume, SpecificHeatConstantVolumeTemperatureFunction},
        {ThermodynamicProperty::SpecificHeatConstantPressure, SpecificHeatConstantPressureTemperatureFunction},
        {ThermodynamicProperty::SpeedOfSound, SpeedOfSoundTemperatureFunction},
        {ThermodynamicProperty::SpeciesSensibleEnthalpy, SpeciesSensibleEnthalpyTemperatureFunction}};

   public:
    explicit PerfectGas(std::shared_ptr<ablate::parameters::Parameters>, std::vector<std::string> species = {});
    void View(std::ostream& stream) const override;
    DecodeStateFunction GetDecodeStateFunction() override { return PerfectGasDecodeState; }
    void* GetDecodeStateContext() override { return &parameters; }
    ComputeTemperatureFunction GetComputeTemperatureFunction() override { return PerfectGasComputeTemperature; }
    void* GetComputeTemperatureContext() override { return &parameters; }
    ComputeSpeciesSensibleEnthalpyFunction GetComputeSpeciesSensibleEnthalpyFunction() override { return PerfectGasComputeSpeciesSensibleEnthalpy; }
    void* GetComputeSpeciesSensibleEnthalpyContext() override { return &parameters; }
    ComputeDensityFunctionFromTemperaturePressure GetComputeDensityFunctionFromTemperaturePressureFunction() override { return PerfectGasComputeDensityFunctionFromTemperaturePressure; }
    void* GetComputeDensityFunctionFromTemperaturePressureContext() override { return &parameters; }
    ComputeSensibleInternalEnergyFunction GetComputeSensibleInternalEnergyFunction() override { return PerfectGasComputeSensibleInternalEnergy; }
    void* GetComputeSensibleInternalEnergyContext() override { return &parameters; }
    ComputeSensibleEnthalpyFunction GetComputeSensibleEnthalpyFunction() override { return PerfectGasComputeSensibleEnthalpy; }
    void* GetComputeSensibleEnthalpyContext() override { return &parameters; }
    ComputeSpecificHeatFunction GetComputeSpecificHeatConstantPressureFunction() override { return PerfectGasComputeSpecificHeatConstantPressure; }
    void* GetComputeSpecificHeatConstantPressureContext() override { return &parameters; }
    ComputeSpecificHeatFunction GetComputeSpecificHeatConstantVolumeFunction() override { return PerfectGasComputeSpecificHeatConstantVolume; }
    void* GetComputeSpecificHeatConstantVolumeContext() override { return &parameters; }
    PetscReal GetSpecificHeatRatio() const { return parameters.gamma; }
    PetscReal GetGasConstant() const { return parameters.rGas; }

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields
     * @param property
     * @param fields
     * @return
     */
    ThermodynamicFunction GetThermodynamicFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override;
    ;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields and temperature
     * @param property
     * @param fields
     * @return
     */
    ThermodynamicTemperatureFunction GetThermodynamicTemperatureFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override;
    ;

    const std::vector<std::string>& GetSpecies() const override { return species; }
};

}  // namespace ablate::eos
#endif  // ABLATELIBRARY_PERFECTGAS_HPP
