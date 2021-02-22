#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "parameters/factoryParameters.hpp"
#include "parser/mockFactory.hpp"
#include <memory>
using ::testing::AtLeast;

namespace ablateTesting::parameters {

using namespace ablate::parameters;

TEST(FactoryParameterTests, ShouldCreateFactoryParameters) {
    // arrange
    std::shared_ptr<ablate::parser::Factory> mockFactory = std::make_shared<ablateTesting::parser::MockFactory>();

    // act
    auto factoryParameters = std::make_shared<ablate::parameters::FactoryParameters>(mockFactory);

    // assert
    ASSERT_TRUE(factoryParameters != nullptr);
}

TEST(FactoryParameterTests, ShouldCreateFromRegistar) {
    // arrange
    std::shared_ptr<ablateTesting::parser::MockFactory> mockFactory = std::make_shared<ablateTesting::parser::MockFactory>();
    const std::string expectedClassType = "";
    EXPECT_CALL(*mockFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedClassType));

    // act
    auto instance = ResolveAndCreate<ablate::parameters::Parameters>(mockFactory);

    // assert
    ASSERT_TRUE(instance != nullptr) << " should create an instance of the Parameters";
    ASSERT_TRUE(std::dynamic_pointer_cast<FactoryParameters>(instance) != nullptr) << " should be an instance of FactoryParameters";
}

}