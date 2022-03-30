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

    static PetscErrorCode StiffenedGasDecodeState(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy,
                                                  PetscReal* a, PetscReal* p, void* ctx);
    static PetscErrorCode StiffenedGasComputeTemperature(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx);

    static PetscErrorCode StiffenedGasComputeSpeciesSensibleEnthalpy(PetscReal T, PetscReal* hi, void* ctx);

    static PetscErrorCode StiffenedGasComputeDensityFunctionFromTemperaturePressure(PetscReal T, PetscReal pressure, const PetscReal yi[], PetscReal* density, void* ctx);

    static PetscErrorCode StiffenedGasComputeSensibleInternalEnergy(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleInternalEnergy, void* ctx);

    static PetscErrorCode StiffenedGasComputeSensibleEnthalpy(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx);

    static PetscErrorCode StiffenedGasComputeSpecificHeatConstantPressure(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx);

    static PetscErrorCode StiffenedGasComputeSpecificHeatConstantVolume(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx);

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
    ThermodynamicFunction GetThermodynamicFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override{
        return ThermodynamicFunction{};
    };

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields and temperature
     * @param property
     * @param fields
     * @return
     */
    ThermodynamicTemperatureFunction GetThermodynamicTemperatureFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override{
        return ThermodynamicTemperatureFunction{};
    };

    const std::vector<std::string>& GetSpecies() const override { return species; }
};

}  // namespace ablate::eos

#endif  // ABLATELIBRARY_STIFFENEDGAS_HPP
