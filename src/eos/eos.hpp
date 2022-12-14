#ifndef ABLATELIBRARY_EOS_HPP
#define ABLATELIBRARY_EOS_HPP
#include <petsc.h>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "domain/field.hpp"

namespace ablate::eos {

enum class ThermodynamicProperty {
    Pressure,
    Temperature,
    InternalSensibleEnergy,
    SensibleEnthalpy,
    SpecificHeatConstantVolume,
    SpecificHeatConstantPressure,
    SpeedOfSound,
    SpeciesSensibleEnthalpy,
    Density
};

/**
 * Simple struct representing the context and function for computing any thermodynamic value when temperature is not available.
 */
struct ThermodynamicFunction {
    PetscErrorCode (*function)(const PetscReal conserved[], PetscReal* property, void* ctx) = nullptr;
    std::shared_ptr<void> context = nullptr;
};

/**
 * Simple struct representing the context and function for computing any thermodynamic value when temperature is available.
 */
struct ThermodynamicTemperatureFunction {
    PetscErrorCode (*function)(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx) = nullptr;
    std::shared_ptr<void> context = nullptr;
};

/**
 * Simple function representing the context and function for computing a field from two specified properties, velocity, and Yi
 */
using FieldFunction = std::function<void(PetscReal property1, PetscReal property2, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[])>;

/**
 * The equation of state is designed to to compute thermodynamic properties based upon the conserved field variables being solved.  This can range from euler & densityYi and  euler & progresses
 * variables with ChemTab.
 */
class EOS {
   protected:
    const std::string type;

   public:
    explicit EOS(std::string typeIn) : type(std::move(typeIn)){};
    virtual ~EOS() = default;

    /**
     * Print the details of this eos
     * @param stream
     */
    virtual void View(std::ostream& stream) const = 0;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] virtual ThermodynamicFunction GetThermodynamicFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const = 0;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields and temperature
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] virtual ThermodynamicTemperatureFunction GetThermodynamicTemperatureFunction(ThermodynamicProperty property, const std::vector<domain::Field>& fields) const = 0;

    /**
     * Single function to produce fieldFunction function for any two properties, velocity, and species mass fractions.  These calls can be slower and should be used for init/output only
     * @param field
     * @param property1
     * @param property2
     */
    [[nodiscard]] virtual FieldFunction GetFieldFunctionFunction(const std::string& field, ThermodynamicProperty property1, ThermodynamicProperty property2) const = 0;

    /**
     * Required species to utilize the equation of state
     * species model functions
     * @return
     */
    [[nodiscard]] virtual const std::vector<std::string>& GetSpeciesVariables() const = 0;

    /**
     * Returns a vector of all extra variables required to utilize the equation of state
     * @return
     */
    [[nodiscard]] virtual const std::vector<std::string>& GetProgressVariables() const = 0;

    /**
     * Species known by this equation of state.  This list is used for the FieldFunction calculations. This can be the same as the GetSpeciesVariables.
     * species model functions
     * @return
     */
    [[nodiscard]] virtual const std::vector<std::string>& GetSpecies() const { return GetSpeciesVariables(); }

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
        case ThermodynamicProperty::Density:
            return "density";
    }
    return "";
}

/**
 * support function to get thermodynamic string name
 * @param prop
 * @return
 */
constexpr ThermodynamicProperty from_string(const std::string_view& prop) {
    if (prop == "pressure") return ThermodynamicProperty::Pressure;
    if (prop == "temperature") return ThermodynamicProperty::Temperature;
    if (prop == "internalSensibleEnergy") return ThermodynamicProperty::InternalSensibleEnergy;
    if (prop == "sensibleEnthalpy") return ThermodynamicProperty::SensibleEnthalpy;
    if (prop == "specificHeatConstantVolume") return ThermodynamicProperty::SpecificHeatConstantVolume;
    if (prop == "specificHeatConstantPressure") return ThermodynamicProperty::SpecificHeatConstantPressure;
    if (prop == "speedOfSound") return ThermodynamicProperty::SpeedOfSound;
    if (prop == "speciesSensibleEnthalpy") return ThermodynamicProperty::SpeciesSensibleEnthalpy;
    if (prop == "density") return ThermodynamicProperty::Density;
    throw std::invalid_argument("No known ThermodynamicProperty for " + std::string(prop));
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

/**
 * Support function for printing a thermodynamic property
 * @param out
 * @param prop
 * @return
 */
inline std::istream& operator>>(std::istream& in, ThermodynamicProperty& prop) {
    std::string propString;
    in >> propString;
    prop = from_string(propString);
    return in;
}

}  // namespace ablate::eos

#endif  // ABLATELIBRARY_EOS_HPP
