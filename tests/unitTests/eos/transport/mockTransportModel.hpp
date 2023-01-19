#ifndef ABLATELIBRARY_MOCKTRANSPORTMODEL_HPP
#define ABLATELIBRARY_MOCKTRANSPORTMODEL_HPP

#include <functional>
#include <map>
#include <ostream>
#include <string>
#include "eos/transport/transportModel.hpp"
#include "gmock/gmock.h"

namespace ablateTesting::eos::transport {

class MockTransportModel : public ablate::eos::transport::TransportModel {
   public:
    static PetscErrorCode MockThermodynamicFunction(const PetscReal conserved[], PetscReal* property, void* ctx) {
        auto function = (std::function<PetscErrorCode(const PetscReal[], PetscReal*)>*)ctx;
        (*function)(conserved, property);
        return 0;
    }

    static PetscErrorCode MockThermodynamicTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx) {
        auto function = (std::function<PetscErrorCode(const PetscReal[], PetscReal T, PetscReal*)>*)ctx;
        (*function)(conserved, temperature, property);
        return 0;
    }

   public:
    MockTransportModel() {}

    MOCK_METHOD(ablate::eos::ThermodynamicFunction, GetTransportFunction, (ablate::eos::transport::TransportProperty, const std::vector<ablate::domain::Field>&), (const, override));
    MOCK_METHOD(ablate::eos::ThermodynamicTemperatureFunction, GetTransportTemperatureFunction, (ablate::eos::transport::TransportProperty, const std::vector<ablate::domain::Field>&), (const, override));

    static ablate::eos::ThermodynamicFunction CreateMockThermodynamicFunction(std::function<void(const PetscReal[], PetscReal*)> function) {
        return ablate::eos::ThermodynamicFunction{.function = MockThermodynamicFunction, .context = std::make_shared<std::function<void(const PetscReal[], PetscReal*)>>(function)};
    }

    static ablate::eos::ThermodynamicTemperatureFunction CreateMockThermodynamicTemperatureFunction(std::function<void(const PetscReal[], PetscReal temperature, PetscReal*)> function) {
        return ablate::eos::ThermodynamicTemperatureFunction{.function = MockThermodynamicTemperatureFunction,
                                                             .context = std::make_shared<std::function<void(const PetscReal[], PetscReal temperature, PetscReal*)>>(function)};
    }
};
}  // namespace ablateTesting::eos

#endif  // ABLATELIBRARY_MOCKTRANSPORTMODEL_HPP
