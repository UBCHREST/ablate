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
}  // namespace ablate::eos::transport
#endif  // ABLATELIBRARY_TRANSPORTMODEL_HPP
