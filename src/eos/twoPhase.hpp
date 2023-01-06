#ifndef ABLATELIBRARY_TWOPHASE_HPP
#define ABLATELIBRARY_TWOPHASE_HPP

#include <memory>
#include "eos.hpp"
#include "eos/perfectGas.hpp"
#include "eos/stiffenedGas.hpp"
#include "parameters/parameters.hpp"
namespace ablate::eos {
class TwoPhase : public EOS { // , public std::enabled_shared_from_this<TwoPhase>
   private:
    const std::shared_ptr<eos::EOS> eos1;
    const std::shared_ptr<eos::EOS> eos2;
    // this mixed eos does not allow species, get species from eos1 eos2
    const std::vector<std::string> species;
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
    // bunch of functions here
    static PetscErrorCode DensityFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode PressureFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode TemperatureFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode InternalSensibleEnergyFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SensibleEnthalpyFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantVolumeFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantPressureFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpeedOfSoundFunction(const PetscReal conserved[], PetscReal* property, void* ctx);
    static PetscErrorCode SpeciesSensibleEnthalpyFunction(const PetscReal conserved[], PetscReal* property, void* ctx);

    static PetscErrorCode DensityTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode PressureTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode TemperatureTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode InternalSensibleEnergyTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SensibleEnthalpyTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantVolumeTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpecificHeatConstantPressureTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpeedOfSoundTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    static PetscErrorCode SpeciesSensibleEnthalpyTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);

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
    explicit TwoPhase(std::shared_ptr<eos::EOS> eos1, std::shared_ptr<eos::EOS> eos2);
    void View(std::ostream& stream) const override;

    ThermodynamicFunction GetThermodynamicFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override;

    ThermodynamicTemperatureFunction GetThermodynamicTemperatureFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override;

    EOSFunction GetFieldFunctionFunction(const std::string& field, ThermodynamicProperty property1, ThermodynamicProperty property2, std::vector<std::string> otherProperties) const override;

    const std::vector<std::string>& GetSpeciesVariables() const override { return species; } // need to modify this
    [[nodiscard]] virtual const std::vector<std::string>& GetProgressVariables() const override { return ablate::utilities::VectorUtilities::Empty<std::string>; }
};
} // namespace ablate::eos

#endif  // ABLATELIBRARY_TWOPHASE_HPP
