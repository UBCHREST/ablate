#include "gtest/gtest.h"
#include "parameters/mapParameters.hpp"

TEST(MapParametersTests, ShouldCreateFromMap) {
    // arrange
    std::map<std::string, std::string> map = {{"item1", "value1"}, {"item2", "value2"}};

    // act
    ablate::parameters::MapParameters mapParameters(map);

    // assert
    ASSERT_EQ(mapParameters.GetString("item1"), "value1");
    ASSERT_EQ(mapParameters.GetString("item2"), "value2");
}

TEST(MapParametersTests, ShouldCreateFromInitializerList) {
    // arrange
    // act
    ablate::parameters::MapParameters mapParameters = {{"item1", "value1"}, {"item2", "value2"}, {"item3", 234}};

    // assert
    ASSERT_EQ(mapParameters.GetString("item1"), "value1");
    ASSERT_EQ(mapParameters.GetString("item2"), "value2");
    ASSERT_EQ(mapParameters.GetString("item3"), "234");
}

TEST(MapParametersTests, ShouldSupportCreateFunction) {
    // arrange
    // act
    auto mapParameters = ablate::parameters::MapParameters::Create({{"item1", "value1"}, {"item2", "value2"}, {"item3", 234}});

    // assert
    ASSERT_EQ(mapParameters->GetString("item1"), "value1");
    ASSERT_EQ(mapParameters->GetString("item2"), "value2");
    ASSERT_EQ(mapParameters->GetString("item3"), "234");
}