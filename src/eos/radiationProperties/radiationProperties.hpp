#ifndef ABLATELIBRARY_RADIATIONMODEL_H
#define ABLATELIBRARY_RADIATIONMODEL_H

#include <petscsystypes.h>
#include "eos/eos.hpp"

namespace ablate::eos::radiationProperties {

enum class RadiationProperty { Absorptivity, Emissivity };

class RadiationModel {
   public:
    virtual ~RadiationModel() = default;

    /**
     * Function to produce radiation absorption properties based upon the available fields
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] virtual ThermodynamicFunction GetRadiationPropertiesFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const = 0;

    /**
     * Function to produce radiation absorption properties based upon the available fields and temperature
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] virtual ThermodynamicTemperatureFunction GetRadiationPropertiesTemperatureFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const = 0;
};

/**
 * support function to get RadiationProperty string name
 * @param prop
 * @return
 */
constexpr std::string_view to_string(const RadiationProperty& prop) {
    switch (prop) {
        case RadiationProperty::Absorptivity:
            return "absorptivity";
        case RadiationProperty::Emissivity:
            return "emissivity";
    }
    return "";
}

/**
 * support function to get RadiationProperty string name
 * @param prop
 * @return
 */
constexpr RadiationProperty from_string(const std::string_view& prop) {
    if (prop == "absorptivity") return RadiationProperty::Absorptivity;
    if (prop == "emissivity") return RadiationProperty::Emissivity;
    throw std::invalid_argument("No known RadiationProperty for " + std::string(prop));
}

/**
 * Support function for printing a RadiationProperty property
 * @param out
 * @param prop
 * @return
 */
inline std::ostream& operator<<(std::ostream& out, const RadiationProperty& prop) {
    auto string = to_string(prop);
    out << string;
    return out;
}

/**
 * Support function for printing a RadiationProperty property
 * @param out
 * @param prop
 * @return
 */
inline std::istream& operator>>(std::istream& in, RadiationProperty& prop) {
    std::string propString;
    in >> propString;
    prop = from_string(propString);
    return in;
}

}  // namespace ablate::eos::radiationProperties

#endif  // ABLATELIBRARY_RADIATIONMODEL_H