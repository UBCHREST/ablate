#ifndef ABLATELIBRARY_TRANSPORTMODEL_HPP
#define ABLATELIBRARY_TRANSPORTMODEL_HPP

#include <petscsystypes.h>
#include "eos/eos.hpp"
namespace ablate::eos::transport {

enum class TransportProperty { Conductivity, Viscosity, Diffusivity };

class TransportModel {
   public:
    virtual ~TransportModel() = default;

    /**
     * Single function to produce transport function for any property based upon the available fields
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] virtual ThermodynamicFunction GetTransportFunction(TransportProperty property, const std::vector<domain::Field>& fields) const = 0;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields and temperature
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] virtual ThermodynamicTemperatureFunction GetTransportTemperatureFunction(TransportProperty property, const std::vector<domain::Field>& fields) const = 0;
};

/**
 * support function to get transportProperty string name
 * @param prop
 * @return
 */
constexpr std::string_view to_string(const TransportProperty& prop) {
    switch (prop) {
        case TransportProperty::Conductivity:
            return "conductivity";
        case TransportProperty::Viscosity:
            return "viscosity";
        case TransportProperty::Diffusivity:
            return "diffusivity";
    }
    return "";
}

/**
 * support function to get transportProperty string name
 * @param prop
 * @return
 */
constexpr TransportProperty from_string(const std::string_view& prop) {
    if (prop == "conductivity") return TransportProperty::Conductivity;
    if (prop == "viscosity") return TransportProperty::Viscosity;
    if (prop == "diffusivity") return TransportProperty::Diffusivity;
    throw std::invalid_argument("No known TransportProperty for " + std::string(prop));
}

/**
 * Support function for printing a transportProperty property
 * @param out
 * @param prop
 * @return
 */
inline std::ostream& operator<<(std::ostream& out, const TransportProperty& prop) {
    auto string = to_string(prop);
    out << string;
    return out;
}

/**
 * Support function for printing a transportProperty property
 * @param out
 * @param prop
 * @return
 */
inline std::istream& operator>>(std::istream& in, TransportProperty& prop) {
    std::string propString;
    in >> propString;
    prop = from_string(propString);
    return in;
}

}  // namespace ablate::eos::transport
#endif  // ABLATELIBRARY_TRANSPORTMODEL_HPP
