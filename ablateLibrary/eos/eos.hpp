#ifndef ABLATELIBRARY_EOS_HPP
#define ABLATELIBRARY_EOS_HPP
#include <petsc.h>
#include <iostream>
#include <string>
#include <vector>
#include "domain/field.hpp"

namespace ablate::eos {

enum class ThermodynamicProperty { Pressure, Temperature, InternalSensibleEnergy, SensibleEnthalpy, SpecificHeatConstantVolume, SpecificHeatConstantPressure, SpeedOfSound, SpeciesSensibleEnthalpy };

/**
 * The internalEnergy computed is without the enthalpy of formation of the species.
 */
using DecodeStateFunction = PetscErrorCode (*)(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a,
                                               PetscReal* p, void* ctx);
using ComputeTemperatureFunction = PetscErrorCode (*)(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx);

using ComputeSensibleInternalEnergyFunction = PetscErrorCode (*)(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleInternalEnergy, void* ctx);

using ComputeSensibleEnthalpyFunction = PetscErrorCode (*)(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx);

using ComputeSpecificHeatFunction = PetscErrorCode (*)(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx);

using ComputeSpeciesSensibleEnthalpyFunction = PetscErrorCode (*)(PetscReal T, PetscReal* hi, void* ctx);

using ComputeDensityFunctionFromTemperaturePressure = PetscErrorCode (*)(PetscReal T, PetscReal pressure, const PetscReal yi[], PetscReal* density, void* ctx);

/**
 * Simple struct representing the context and function for computing any thermodynamic value when temperature is not available.
 */
struct ThermodynamicFunction {
    PetscErrorCode (*function)(const PetscReal conserved[], PetscReal* property, void* ctx);
    std::shared_ptr<void> context;
};

/**
 * Simple struct representing the context and function for computing any thermodynamic value when temperature is available.
 */
struct ThermodynamicTemperatureFunction {
    PetscErrorCode (*function)(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);
    std::shared_ptr<void> context;
};

/**
 * The EOS is a combination of species model and EOS.  This allows the eos to dictate the order/number of species.  This can be relaxed in the future
 */
class EOS {
   protected:
    const std::string type;

   public:
    EOS(std::string typeIn) : type(typeIn){};
    virtual ~EOS() = default;

    /**
     * Print the details of this eos
     * @param stream
     */
    virtual void View(std::ostream& stream) const = 0;

    virtual DecodeStateFunction GetDecodeStateFunction() = 0;
    virtual void* GetDecodeStateContext() = 0;
    virtual ComputeTemperatureFunction GetComputeTemperatureFunction() = 0;
    virtual void* GetComputeTemperatureContext() = 0;
    virtual ComputeSpeciesSensibleEnthalpyFunction GetComputeSpeciesSensibleEnthalpyFunction() = 0;
    virtual void* GetComputeSpeciesSensibleEnthalpyContext() = 0;
    virtual ComputeDensityFunctionFromTemperaturePressure GetComputeDensityFunctionFromTemperaturePressureFunction() = 0;
    virtual void* GetComputeDensityFunctionFromTemperaturePressureContext() = 0;
    virtual ComputeSensibleInternalEnergyFunction GetComputeSensibleInternalEnergyFunction() = 0;
    virtual void* GetComputeSensibleInternalEnergyContext() = 0;
    virtual ComputeSensibleEnthalpyFunction GetComputeSensibleEnthalpyFunction() = 0;
    virtual void* GetComputeSensibleEnthalpyContext() = 0;
    virtual ComputeSpecificHeatFunction GetComputeSpecificHeatConstantPressureFunction() = 0;
    virtual void* GetComputeSpecificHeatConstantPressureContext() = 0;
    virtual ComputeSpecificHeatFunction GetComputeSpecificHeatConstantVolumeFunction() = 0;
    virtual void* GetComputeSpecificHeatConstantVolumeContext() = 0;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields and temperature
     * @param property
     * @param fields
     * @return
     */
    virtual ThermodynamicTemperatureFunction GetThermodynamicTemperatureFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const = 0;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields
     * @param property
     * @param fields
     * @return
     */
    virtual ThermodynamicFunction GetThermodynamicFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const = 0;

    /**
     * Species supported by this EOS
     * species model functions
     * @return
     */
    virtual const std::vector<std::string>& GetSpecies() const = 0;

    /**
     * Support function for printing any eos
     * @param out
     * @param eos
     * @return
     */
    friend std::ostream& operator<<(std::ostream& out, const EOS& eos) {
        eos.View(out);
        return out;
    }
};

/**
 * support function to get thermodynamic string name
 * @param prop
 * @return
 */
constexpr std::string_view to_string(const ThermodynamicProperty& prop) {
    switch (prop) {
        case ThermodynamicProperty::Pressure:
            return "pressure";
        case ThermodynamicProperty::Temperature:
            return "temperature";
        case ThermodynamicProperty::InternalSensibleEnergy:
            return "internalSensibleEnergy";
        case ThermodynamicProperty::SensibleEnthalpy:
            return "sensibleEnthalpy";
        case ThermodynamicProperty::SpecificHeatConstantVolume:
            return "specificHeatConstantVolume";
        case ThermodynamicProperty::SpecificHeatConstantPressure:
            return "specificHeatConstantPressure";
        case ThermodynamicProperty::SpeedOfSound:
            return "speedOfSound";
        case ThermodynamicProperty::SpeciesSensibleEnthalpy:
            return "speciesSensibleEnthalpy";
    }
    return "";
}

/**
 * Support function for printing a thermodynamic property
 * @param out
 * @param prop
 * @return
 */
inline std::ostream& operator<<(std::ostream& out, const ThermodynamicProperty& prop) {
    auto string = to_string(prop);
    out << string;
    return out;
}

}  // namespace ablate::eos

#endif  // ABLATELIBRARY_EOS_HPP
