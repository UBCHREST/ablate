#ifndef ABLATELIBRARY_MOCKPARAMETERS_HPP
#define ABLATELIBRARY_MOCKPARAMETERS_HPP
#include "parameters/parameters.hpp"
#include "gmock/gmock.h"

namespace ablateTesting::parameters {

class MockParameters : public ablate::parameters::Parameters {
   public:
    MOCK_METHOD(std::optional<std::string>, GetString, (std::string paramName), (const, override));
};

}
#endif  // ABLATELIBRARY_MOCKPARAMETERS_HPP
