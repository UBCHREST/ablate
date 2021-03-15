#ifndef ABLATELIBRARY_MOCKPARAMETERS_HPP
#define ABLATELIBRARY_MOCKPARAMETERS_HPP
#include "gmock/gmock.h"
#include "parameters/parameters.hpp"

namespace ablateTesting::parameters {

class MockParameters : public ablate::parameters::Parameters {
   public:
    MOCK_METHOD(std::optional<std::string>, GetString, (std::string paramName), (const, override));
};

}  // namespace ablateTesting::parameters
#endif  // ABLATELIBRARY_MOCKPARAMETERS_HPP
