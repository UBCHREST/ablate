#ifndef mpitestparamfixture_h
#define mpitestparamfixture_h
#include <gtest/gtest.h>

#include "MpiTestFixture.hpp"

using namespace testingResources;
namespace testingResources {

class MpiTestParamFixture : public MpiTestFixture, public ::testing::WithParamInterface<MpiTestParameter> {
   protected:
    void SetUp() override { SetMpiParameters(GetParam()); }
};
}  // namespace testingResources
#endif