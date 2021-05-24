#include "gtest/gtest.h"
#include "eos.h"
#include "PetscTestFixture.hpp"
#include "PetscTestViewer.hpp"

struct EOSTestParameters {
    std::string eosType;
    std::map<std::string, std::string> options;
    std::string expectedView;
};

class EOSTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<EOSTestParameters>  {
};

TEST_P(EOSTestFixture, ShouldCreateAndView) {

    // arrange
    EOSData eos;
    EOSCreate(&eos) >> errorChecker;

    // If provided, set the eosType
    if(!GetParam().eosType.empty()){
        EOSSetType(eos, GetParam().eosType.c_str()) >> errorChecker;
    }

    // Define a set of options
    PetscOptions options;
    PetscOptionsCreate(&options) >> errorChecker;
    // add each option in the map
    for(const auto& option: GetParam().options){
        PetscOptionsSetValue(options, option.first.c_str(), option.second.c_str()) >> errorChecker;
    }

    // setup the eos from options
    EOSSetOptions(eos, options) >> errorChecker;
    EOSSetFromOptions(eos) >> errorChecker;

    // setup a test viewer
    testingResources::PetscTestViewer viewer;

    // act
    EOSView(eos, viewer.GetViewer());

    // assert the output is as expected
    auto outputString = viewer.GetString();
    ASSERT_EQ(outputString, GetParam().expectedView);

    // cleanup
    EOSDestroy(&eos) >> errorChecker;
}

INSTANTIATE_TEST_SUITE_P(
    EOSTests, EOSTestFixture,
    testing::Values(
    (EOSTestParameters){
            .eosType = "perfectGas",
            .options = {},
            .expectedView = "EOS: perfectGas\n  gamma: 1.400000\n  Rgas: 287.000000\n"
        },
    (EOSTestParameters){
        .eosType = "perfectGas",
        .options = {{"-gamma", "3.2"}, {"-Rgas", "100.2"}},
        .expectedView = "EOS: perfectGas\n  gamma: 3.200000\n  Rgas: 100.200000\n"
    }
),[](const testing::TestParamInfo<EOSTestParameters>& info) { return std::to_string(info.index) + "_" + info.param.eosType; });