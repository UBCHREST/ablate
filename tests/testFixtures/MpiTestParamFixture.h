#ifndef mpitestparamfixture_h
#define mpitestparamfixture_h
#include <gtest/gtest.h>
#include "MpiTestFixture.h"

class MpiTestParamFixture : public MpiTestFixture, public ::testing::WithParamInterface<MpiTestParameter> {
protected:
    void SetUp() override{
        SetMpiParameters(GetParam());
    }
};


#endif
