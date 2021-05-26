#ifndef petsctestfixture_h
#define petsctestfixture_h
#include <gtest/gtest.h>
#include <petscsys.h>
#include <filesystem>
#include "MpiTestFixture.hpp"
#include "PetscTestErrorChecker.hpp"

namespace testingResources {

/*Simple test fixture to setup petsc without any input arguments.  This is a reused environment and should not be used with tests that use arguments.*/
class PetscTestFixture : public ::testing::Test {
   protected:
    PetscTestErrorChecker errorChecker;
    void SetUp() override {
        PetscBool petsInitialized;
        PetscInitialized(&petsInitialized) >> errorChecker;

        if (!petsInitialized) {
            PetscInitializeNoArguments() >> errorChecker;
        }
    }
};

};      // namespace testingResources
#endif  // mpitestfixture_h