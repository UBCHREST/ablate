#ifndef ABLATELIBRARY_MOCKEOS_HPP
#define ABLATELIBRARY_MOCKEOS_HPP

#include <functional>
#include <map>
#include <ostream>
#include <string>
#include "eos/eos.hpp"
#include "gmock/gmock.h"

namespace ablateTesting::eos {

class MockEOS : public ablate::eos::EOS {
   private:
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
    MockEOS() : ablate::eos::EOS("MockEOS") {}

    MOCK_METHOD(void, View, (std::ostream & stream), (override, const));
    MOCK_METHOD(const std::vector<std::string>&, GetSpeciesVariables, (), (const, override));
    MOCK_METHOD(const std::vector<std::string>&, GetProgressVariables, (), (const, override));

    MOCK_METHOD(ablate::eos::ThermodynamicFunction, GetThermodynamicFunction, (ablate::eos::ThermodynamicProperty, const std::vector<ablate::domain::Field>&), (const, override));
    MOCK_METHOD(ablate::eos::ThermodynamicTemperatureFunction, GetThermodynamicTemperatureFunction, (ablate::eos::ThermodynamicProperty, const std::vector<ablate::domain::Field>&), (const, override));
    MOCK_METHOD(ablate::eos::FieldFunction, GetFieldFunctionFunction, (const std::string& field, ablate::eos::ThermodynamicProperty, ablate::eos::ThermodynamicProperty), (const, override));

    static ablate::eos::ThermodynamicFunction CreateMockThermodynamicFunction(std::function<void(const PetscReal[], PetscReal*)> function) {
        return ablate::eos::ThermodynamicFunction{.function = MockThermodynamicFunction, .context = std::make_shared<std::function<void(const PetscReal[], PetscReal*)>>(function)};
    }

    static ablate::eos::ThermodynamicTemperatureFunction CreateMockThermodynamicTemperatureFunction(std::function<void(const PetscReal[], PetscReal temperature, PetscReal*)> function) {
        return ablate::eos::ThermodynamicTemperatureFunction{.function = MockThermodynamicTemperatureFunction,
                                                             .context = std::make_shared<std::function<void(const PetscReal[], PetscReal temperature, PetscReal*)>>(function)};
    }
};
}  // namespace ablateTesting::eos

#endif  // ABLATELIBRARY_MOCKEOS_HPP
