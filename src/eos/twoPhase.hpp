#ifndef ABLATELIBRARY_TWOPHASE_HPP
#define ABLATELIBRARY_TWOPHASE_HPP

#include <memory>
#include "eos.hpp"
#include "eos/perfectGas.hpp"
#include "eos/stiffenedGas.hpp"
#include "parameters/parameters.hpp"
namespace ablate::eos {
class TwoPhase : public EOS {  // , public std::enabled_shared_from_this<TwoPhase>
   public:
    inline const static std::string VF = "volumeFraction";

   private:
    const std::shared_ptr<eos::EOS> eos1;
    const std::shared_ptr<eos::EOS> eos2;
    // this mixed eos does not allow species, get species from eos1 eos2
    std::vector<std::string> species;
    std::vector<std::string> otherPropertiesList = {"VF"};  // otherProperties must include volumeFraction for Field Function initialization
    struct Parameters {
        PetscReal gamma1;
        PetscReal rGas1;
        PetscReal Cp1;
        PetscReal p01;
        PetscReal gamma2;
        PetscReal rGas2;
        PetscReal Cp2;
        PetscReal p02;
        PetscInt numberSpecies1;
        std::vector<std::string> species1;
        PetscInt numberSpecies2;
        std::vector<std::string> species2;
    };
    Parameters parameters;
    struct FunctionContext {
        PetscInt dim;
        PetscInt eulerOffset;
        PetscInt densityVFOffset;
        PetscInt volumeFractionOffset;
        Parameters parameters;
    };

   public:
    struct DecodeIn {
        PetscReal alpha;
        PetscReal alphaRho1;
        PetscReal rho;
        PetscReal e;
        Parameters parameters;
    };
    struct DecodeOut {
        PetscReal rho1;
        PetscReal rho2;
        PetscReal e1;
        PetscReal e2;
        PetscReal p;
        PetscReal T;
    };

   private:
    // functions for all cases
    static PetscErrorCode DensityFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode InternalSensibleEnergyFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpeciesSensibleEnthalpyFunction(const PetscReal conserved[], PetscReal* property, void* ctx);

    static PetscErrorCode DensityTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode InternalSensibleEnergyTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpeciesSensibleEnthalpyTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);

    // functions for GasGas case
    static PetscErrorCode PressureFunctionGasGas(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode TemperatureFunctionGasGas(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SensibleEnthalpyFunctionGasGas(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantVolumeFunctionGasGas(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantPressureFunctionGasGas(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpeedOfSoundFunctionGasGas(const PetscReal conserved[], PetscReal* property, void* ctx);

    static PetscErrorCode PressureTemperatureFunctionGasGas(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode TemperatureTemperatureFunctionGasGas(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SensibleEnthalpyTemperatureFunctionGasGas(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantVolumeTemperatureFunctionGasGas(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantPressureTemperatureFunctionGasGas(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpeedOfSoundTemperatureFunctionGasGas(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);

    // functions for GasLiquid
    static PetscErrorCode PressureFunctionGasLiquid(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode TemperatureFunctionGasLiquid(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SensibleEnthalpyFunctionGasLiquid(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantVolumeFunctionGasLiquid(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantPressureFunctionGasLiquid(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpeedOfSoundFunctionGasLiquid(const PetscReal conserved[], PetscReal* property, void* ctx);

    static PetscErrorCode PressureTemperatureFunctionGasLiquid(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode TemperatureTemperatureFunctionGasLiquid(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SensibleEnthalpyTemperatureFunctionGasLiquid(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantVolumeTemperatureFunctionGasLiquid(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantPressureTemperatureFunctionGasLiquid(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpeedOfSoundTemperatureFunctionGasLiquid(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);

    // functions for LiquidLiquid
    static PetscErrorCode PressureFunctionLiquidLiquid(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode TemperatureFunctionLiquidLiquid(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SensibleEnthalpyFunctionLiquidLiquid(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantVolumeFunctionLiquidLiquid(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantPressureFunctionLiquidLiquid(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpeedOfSoundFunctionLiquidLiquid(const PetscReal conserved[], PetscReal* property, void* ctx);

    static PetscErrorCode PressureTemperatureFunctionLiquidLiquid(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode TemperatureTemperatureFunctionLiquidLiquid(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SensibleEnthalpyTemperatureFunctionLiquidLiquid(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantVolumeTemperatureFunctionLiquidLiquid(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantPressureTemperatureFunctionLiquidLiquid(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpeedOfSoundTemperatureFunctionLiquidLiquid(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);

    using ThermodynamicStaticFunction = PetscErrorCode (*)(const PetscReal conserved[], PetscReal* property, void* ctx);
    using ThermodynamicTemperatureStaticFunction = PetscErrorCode (*)(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);
    // map for GasGas case
    std::map<ThermodynamicProperty, std::pair<ThermodynamicStaticFunction, ThermodynamicTemperatureStaticFunction>> thermodynamicFunctionsGasGas = {
        {ThermodynamicProperty::Density, {DensityFunction, DensityTemperatureFunction}},
        {ThermodynamicProperty::Pressure, {PressureFunctionGasGas, PressureTemperatureFunctionGasGas}},
        {ThermodynamicProperty::Temperature, {TemperatureFunctionGasGas, TemperatureTemperatureFunctionGasGas}},
        {ThermodynamicProperty::InternalSensibleEnergy, {InternalSensibleEnergyFunction, InternalSensibleEnergyTemperatureFunction}},
        {ThermodynamicProperty::SensibleEnthalpy, {SensibleEnthalpyFunctionGasGas, SensibleEnthalpyTemperatureFunctionGasGas}},
        {ThermodynamicProperty::SpecificHeatConstantVolume, {SpecificHeatConstantVolumeFunctionGasGas, SpecificHeatConstantVolumeTemperatureFunctionGasGas}},
        {ThermodynamicProperty::SpecificHeatConstantPressure, {SpecificHeatConstantPressureFunctionGasGas, SpecificHeatConstantPressureTemperatureFunctionGasGas}},
        {ThermodynamicProperty::SpeedOfSound, {SpeedOfSoundFunctionGasGas, SpeedOfSoundTemperatureFunctionGasGas}},
        {ThermodynamicProperty::SpeciesSensibleEnthalpy, {SpeciesSensibleEnthalpyFunction, SpeciesSensibleEnthalpyTemperatureFunction}}};
    // map for GasLiquid case
    std::map<ThermodynamicProperty, std::pair<ThermodynamicStaticFunction, ThermodynamicTemperatureStaticFunction>> thermodynamicFunctionsGasLiquid = {
        {ThermodynamicProperty::Density, {DensityFunction, DensityTemperatureFunction}},
        {ThermodynamicProperty::Pressure, {PressureFunctionGasLiquid, PressureTemperatureFunctionGasLiquid}},
        {ThermodynamicProperty::Temperature, {TemperatureFunctionGasLiquid, TemperatureTemperatureFunctionGasLiquid}},
        {ThermodynamicProperty::InternalSensibleEnergy, {InternalSensibleEnergyFunction, InternalSensibleEnergyTemperatureFunction}},
        {ThermodynamicProperty::SensibleEnthalpy, {SensibleEnthalpyFunctionGasLiquid, SensibleEnthalpyTemperatureFunctionGasLiquid}},
        {ThermodynamicProperty::SpecificHeatConstantVolume, {SpecificHeatConstantVolumeFunctionGasLiquid, SpecificHeatConstantVolumeTemperatureFunctionGasLiquid}},
        {ThermodynamicProperty::SpecificHeatConstantPressure, {SpecificHeatConstantPressureFunctionGasLiquid, SpecificHeatConstantPressureTemperatureFunctionGasLiquid}},
        {ThermodynamicProperty::SpeedOfSound, {SpeedOfSoundFunctionGasLiquid, SpeedOfSoundTemperatureFunctionGasLiquid}},
        {ThermodynamicProperty::SpeciesSensibleEnthalpy, {SpeciesSensibleEnthalpyFunction, SpeciesSensibleEnthalpyTemperatureFunction}}};
    // map for LiquidLiquid case
    std::map<ThermodynamicProperty, std::pair<ThermodynamicStaticFunction, ThermodynamicTemperatureStaticFunction>> thermodynamicFunctionsLiquidLiquid = {
        {ThermodynamicProperty::Density, {DensityFunction, DensityTemperatureFunction}},
        {ThermodynamicProperty::Pressure, {PressureFunctionLiquidLiquid, PressureTemperatureFunctionLiquidLiquid}},
        {ThermodynamicProperty::Temperature, {TemperatureFunctionLiquidLiquid, TemperatureTemperatureFunctionLiquidLiquid}},
        {ThermodynamicProperty::InternalSensibleEnergy, {InternalSensibleEnergyFunction, InternalSensibleEnergyTemperatureFunction}},
        {ThermodynamicProperty::SensibleEnthalpy, {SensibleEnthalpyFunctionLiquidLiquid, SensibleEnthalpyTemperatureFunctionLiquidLiquid}},
        {ThermodynamicProperty::SpecificHeatConstantVolume, {SpecificHeatConstantVolumeFunctionLiquidLiquid, SpecificHeatConstantVolumeTemperatureFunctionLiquidLiquid}},
        {ThermodynamicProperty::SpecificHeatConstantPressure, {SpecificHeatConstantPressureFunctionLiquidLiquid, SpecificHeatConstantPressureTemperatureFunctionLiquidLiquid}},
        {ThermodynamicProperty::SpeedOfSound, {SpeedOfSoundFunctionLiquidLiquid, SpeedOfSoundTemperatureFunctionLiquidLiquid}},
        {ThermodynamicProperty::SpeciesSensibleEnthalpy, {SpeciesSensibleEnthalpyFunction, SpeciesSensibleEnthalpyTemperatureFunction}}};

   public:
    explicit TwoPhase(std::shared_ptr<eos::EOS> eos1, std::shared_ptr<eos::EOS> eos2);
    void View(std::ostream& stream) const override;

    const std::shared_ptr<ablate::eos::EOS> GetEOSGas() const { return eos1; }
    const std::shared_ptr<ablate::eos::EOS> GetEOSLiquid() const { return eos2; }

    ThermodynamicFunction GetThermodynamicFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override;

    ThermodynamicTemperatureFunction GetThermodynamicTemperatureFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override;

    EOSFunction GetFieldFunctionFunction(const std::string& field, ThermodynamicProperty property1, ThermodynamicProperty property2, std::vector<std::string> otherProperties) const override;
    const std::vector<std::string>& GetFieldFunctionProperties() const override { return otherPropertiesList; }  // list of other properties i.e. VF;

    const std::vector<std::string>& GetSpeciesVariables() const override { return species; }  // lists species of eos1 first, then eos2, no distinction for which fluid the species exists in
    [[nodiscard]] virtual const std::vector<std::string>& GetProgressVariables() const override { return ablate::utilities::VectorUtilities::Empty<std::string>; }
};
}  // namespace ablate::eos

#endif  // ABLATELIBRARY_TWOPHASE_HPP
