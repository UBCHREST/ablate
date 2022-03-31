#ifndef ABLATELIBRARY_STIFFENEDGAS_HPP
#define ABLATELIBRARY_STIFFENEDGAS_HPP

#include <memory>
#include "eos.hpp"
#include "parameters/parameters.hpp"
namespace ablate::eos {

class StiffenedGas : public EOS {
   private:
    // the stiffened gas does not allow species
    const std::vector<std::string> species;
    struct Parameters {
        PetscReal gamma;
        PetscReal Cp;
        PetscReal p0;
        PetscInt numberSpecies;
    };
    Parameters parameters;

    struct FunctionContext {
        PetscInt dim;
        PetscInt eulerOffset;
        Parameters parameters;
    };

    static PetscErrorCode StiffenedGasDecodeState(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy,
                                                  PetscReal* a, PetscReal* p, void* ctx);
    static PetscErrorCode StiffenedGasComputeTemperature(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx);

    static PetscErrorCode StiffenedGasComputeSpeciesSensibleEnthalpy(PetscReal T, PetscReal* hi, void* ctx);

    static PetscErrorCode StiffenedGasComputeDensityFunctionFromTemperaturePressure(PetscReal T, PetscReal pressure, const PetscReal yi[], PetscReal* density, void* ctx);

    static PetscErrorCode StiffenedGasComputeSensibleInternalEnergy(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleInternalEnergy, void* ctx);

    static PetscErrorCode StiffenedGasComputeSensibleEnthalpy(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx);

    static PetscErrorCode StiffenedGasComputeSpecificHeatConstantPressure(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx);

    static PetscErrorCode StiffenedGasComputeSpecificHeatConstantVolume(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx);

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
    using ThermodynamicTemperatureStaticFunction = PetscErrorCode (*)(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);
    inline static std::map<ThermodynamicProperty, std::pair<ThermodynamicStaticFunction,ThermodynamicTemperatureStaticFunction>> thermodynamicFunctions =
        {{ThermodynamicProperty::Pressure, {PressureFunction, PressureTemperatureFunction}},
         {ThermodynamicProperty::Temperature, {TemperatureFunction,TemperatureTemperatureFunction}},
         {ThermodynamicProperty::InternalSensibleEnergy, {InternalSensibleEnergyFunction,InternalSensibleEnergyTemperatureFunction}},
         {ThermodynamicProperty::SensibleEnthalpy, {SensibleEnthalpyFunction,SensibleEnthalpyTemperatureFunction}},
         {ThermodynamicProperty::SpecificHeatConstantVolume, {SpecificHeatConstantVolumeFunction,SpecificHeatConstantVolumeTemperatureFunction}},
         {ThermodynamicProperty::SpecificHeatConstantPressure, {SpecificHeatConstantPressureFunction,SpecificHeatConstantPressureTemperatureFunction}},
         {ThermodynamicProperty::SpeedOfSound, {SpeedOfSoundFunction,SpeedOfSoundTemperatureFunction}},
         {ThermodynamicProperty::SpeciesSensibleEnthalpy, {SpeciesSensibleEnthalpyFunction,SpeciesSensibleEnthalpyTemperatureFunction}}};

   public:
    explicit StiffenedGas(std::shared_ptr<ablate::parameters::Parameters>, std::vector<std::string> species = {});
    void View(std::ostream& stream) const override;
    DecodeStateFunction GetDecodeStateFunction() override { return StiffenedGasDecodeState; }
    void* GetDecodeStateContext() override { return &parameters; }
    ComputeTemperatureFunction GetComputeTemperatureFunction() override { return StiffenedGasComputeTemperature; }
    void* GetComputeTemperatureContext() override { return &parameters; }
    ComputeSpeciesSensibleEnthalpyFunction GetComputeSpeciesSensibleEnthalpyFunction() override { return StiffenedGasComputeSpeciesSensibleEnthalpy; }
    void* GetComputeSpeciesSensibleEnthalpyContext() override { return &parameters; }
    ComputeDensityFunctionFromTemperaturePressure GetComputeDensityFunctionFromTemperaturePressureFunction() override { return StiffenedGasComputeDensityFunctionFromTemperaturePressure; }
    void* GetComputeDensityFunctionFromTemperaturePressureContext() override { return &parameters; }
    ComputeSensibleInternalEnergyFunction GetComputeSensibleInternalEnergyFunction() override { return StiffenedGasComputeSensibleInternalEnergy; }
    void* GetComputeSensibleInternalEnergyContext() override { return &parameters; }
    ComputeSensibleEnthalpyFunction GetComputeSensibleEnthalpyFunction() override { return StiffenedGasComputeSensibleEnthalpy; }
    void* GetComputeSensibleEnthalpyContext() override { return &parameters; }
    ComputeSpecificHeatFunction GetComputeSpecificHeatConstantPressureFunction() override { return StiffenedGasComputeSpecificHeatConstantPressure; }
    void* GetComputeSpecificHeatConstantPressureContext() override { return &parameters; }
    ComputeSpecificHeatFunction GetComputeSpecificHeatConstantVolumeFunction() override { return StiffenedGasComputeSpecificHeatConstantVolume; }
    void* GetComputeSpecificHeatConstantVolumeContext() override { return &parameters; }
    PetscReal GetSpecificHeatRatio() const { return parameters.gamma; }
    PetscReal GetSpecificHeatCp() const { return parameters.Cp; }
    PetscReal GetReferencePressure() const { return parameters.p0; }

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


    const std::vector<std::string>& GetSpecies() const override { return species; }
};

}  // namespace ablate::eos

#endif  // ABLATELIBRARY_STIFFENEDGAS_HPP
