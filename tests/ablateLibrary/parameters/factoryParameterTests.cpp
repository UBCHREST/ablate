#include <memory>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "parameters/factoryParameters.hpp"
#include "parser/mockFactory.hpp"
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

TEST(FactoryParameterTests, ShouldGetValuesFromFactory) {
    // arrange
    const std::string paramName = "parm123";

    std::shared_ptr<ablateTesting::parser::MockFactory> mockFactory = std::make_shared<ablateTesting::parser::MockFactory>();
    const std::string expectedClassType = "";
    EXPECT_CALL(*mockFactory, Contains(paramName)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockFactory, Get(ablate::parser::ArgumentIdentifier<std::string>{.inputName = paramName})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("123"));

    auto factoryParameters = std::make_shared<ablate::parameters::FactoryParameters>(mockFactory);

    // act
    auto intValue = factoryParameters->Get<int>(paramName);

    // assert
    ASSERT_EQ(intValue, 123);
}

TEST(FactoryParameterTests, ShouldReturnEmptyValueWhenNotThere) {
    // arrange
    const std::string paramName = "parm123";

    std::shared_ptr<ablateTesting::parser::MockFactory> mockFactory = std::make_shared<ablateTesting::parser::MockFactory>();
    const std::string expectedClassType = "";
    EXPECT_CALL(*mockFactory, Contains(paramName)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));

    auto factoryParameters = std::make_shared<ablate::parameters::FactoryParameters>(mockFactory);

    // act
    auto intValue = factoryParameters->Get<int>(paramName);

    // assert
    ASSERT_FALSE(intValue.has_value());
}

TEST(FactoryParameterTests, ShouldGetAllKeys) {
    // arrange
    std::shared_ptr<ablateTesting::parser::MockFactory> mockFactory = std::make_shared<ablateTesting::parser::MockFactory>();
    std::unordered_set<std::string> expectedKeys = {"key1", "key2"};
    EXPECT_CALL(*mockFactory, GetKeys()).Times(::testing::Exactly(1)).WillOnce(::testing::Return(expectedKeys));

    auto factoryParameters = std::make_shared<ablate::parameters::FactoryParameters>(mockFactory);

    // act
    auto keyValues = factoryParameters->GetKeys();

    // assert
    ASSERT_EQ(expectedKeys, keyValues);
}

}  // namespace ablateTesting::parameters