#ifndef ABLATELIBRARY_MOCKINTERVAL_HPP
#define ABLATELIBRARY_MOCKINTERVAL_HPP

#include <functional>
#include <map>
#include <ostream>
#include <string>
#include "gmock/gmock.h"
#include "io/interval/interval.hpp"

namespace ablateTesting::io::interval {

class MockInterval : public ablate::io::interval::Interval {
   public:
    MockInterval() {}

    /*
     * Mock the only function required for the interval
     */
    MOCK_METHOD(bool, Check, (MPI_Comm comm, PetscInt steps, PetscReal time), (override));
};
}  // namespace ablateTesting::io::interval

#endif  // ABLATELIBRARY_MOCKEOS_HPP
