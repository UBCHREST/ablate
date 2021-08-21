#include <memory>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mockFactory.hpp"
#include "parser/registrar.hpp"

using ::testing::AtLeast;

namespace ablateTesting::parser {

using namespace ablate::parser;

class FactoryMockClass1 {};

TEST(FactoryTests, GetShouldReturnNullPtrWhenOptional) {
    // arrange

    ablate::parser::Registrar<FactoryMockClass1>::Register<FactoryMockClass1>(true, "FactoryMockClass1", "this is a simple mock class");
    auto mockFactory = std::make_shared<MockFactory>();
    EXPECT_CALL(*mockFactory, Contains(std::string("input123"))).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));

    // act
    auto argument = ArgumentIdentifier<FactoryMockClass1>{.inputName = "input123", .optional = true};
    auto result = std::dynamic_pointer_cast<Factory>(mockFactory)->Get(argument);

    // assert
    ASSERT_TRUE(result == nullptr);
}

TEST(FactoryTests, ShouldReturnEmptyListWhenOptional) {
    // arrange

    ablate::parser::Registrar<FactoryMockClass1>::Register<FactoryMockClass1>(true, "FactoryMockClass1", "this is a simple mock class");
    auto mockFactory = std::make_shared<MockFactory>();
    EXPECT_CALL(*mockFactory, Contains(std::string("input123"))).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));

    // act
    auto argument = ArgumentIdentifier<std::vector<FactoryMockClass1>>{.inputName = "input123", .optional = true};
    auto result = std::dynamic_pointer_cast<Factory>(mockFactory)->Get(argument);

    // assert
    ASSERT_TRUE(result.empty());
}

TEST(FactoryTests, ShouldReturnEmptyMapWhenOptional) {
    // arrange

    ablate::parser::Registrar<FactoryMockClass1>::Register<FactoryMockClass1>(true, "FactoryMockClass1", "this is a simple mock class");
    auto mockFactory = std::make_shared<MockFactory>();
    EXPECT_CALL(*mockFactory, Contains(std::string("input123"))).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));

    // act
    auto argument = ArgumentIdentifier<std::map<std::string, FactoryMockClass1>>{.inputName = "input123", .optional = true};
    auto result = std::dynamic_pointer_cast<Factory>(mockFactory)->Get(argument);

    // assert
    ASSERT_TRUE(result.empty());
}

TEST(FactoryTests, GetByNameShouldCallGetWithCorrectArguments) {
    // arrange
    MockFactory mockFactory;
    EXPECT_CALL(mockFactory, Get(ArgumentIdentifier<std::string>{.inputName = "input123"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("result 123"));

    // act
    auto result = mockFactory.GetByName<std::string>("input123");

    // assert
    ASSERT_EQ("result 123", result);
}

TEST(FactoryTests, GetByNameShouldReturnCorrectValue) {
    // arrange
    MockFactory mockFactory;
    EXPECT_CALL(mockFactory, Contains("input123")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(true));
    EXPECT_CALL(mockFactory, Get(ArgumentIdentifier<std::string>{.inputName = "input123"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("result 123"));

    // act
    auto result = mockFactory.GetByName<std::string, std::string>("input123", "default 123");

    // assert
    ASSERT_EQ("result 123", result);
}

TEST(FactoryTests, GetByNameShouldReturnDefaultValue) {
    // arrange
    MockFactory mockFactory;
    EXPECT_CALL(mockFactory, Contains("input123")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));

    // act
    auto result = mockFactory.GetByName<std::string>("input123", std::string("default 123"));

    // assert
    ASSERT_EQ("default 123", result);
}

class DefaultMockClass {
   public:
    std::string name;
    DefaultMockClass(std::string name) : name(name){};
};

TEST(FactoryTests, GetByNameShouldReturnDefaultValueClass) {
    // arrange
    MockFactory mockFactory;
    EXPECT_CALL(mockFactory, Contains("input123")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));

    // act
    auto result = mockFactory.GetByName<DefaultMockClass>("input123", std::make_shared<DefaultMockClass>("default 123"));

    // assert
    ASSERT_EQ("default 123", result->name);
}

TEST(FactoryTests, GetByNameShouldReturnDefaultValueWithList) {
    // arrange
    MockFactory mockFactory;
    EXPECT_CALL(mockFactory, Contains("input123")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));

    // act
    auto result = mockFactory.GetByName<std::vector<DefaultMockClass>>("input123", std::vector<std::shared_ptr<DefaultMockClass>>{std::make_shared<DefaultMockClass>("default 123")});

    // assert
    ASSERT_EQ(1, result.size());
    ASSERT_EQ("default 123", result[0]->name);
}

TEST(FactoryTests, GetByNameShouldReturnDefaultValueWithEmptyList) {
    // arrange
    MockFactory mockFactory;
    EXPECT_CALL(mockFactory, Contains("input123")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));

    // act
    auto result = mockFactory.GetByName<std::vector<DefaultMockClass>>("input123", std::vector<std::shared_ptr<DefaultMockClass>>());

    // assert
    ASSERT_EQ(0, result.size());
}

enum class TestEnum { VECTOR, COMPONENT };

std::istream& operator>>(std::istream& is, TestEnum& v) {
    std::string enumString;
    is >> enumString;

    if (enumString == "vector") {
        v = TestEnum::VECTOR;
    } else if (enumString == "component") {
        v = TestEnum::COMPONENT;
    } else {
        throw std::invalid_argument("Unknown Scope type " + enumString);
    }
    return is;
}

TEST(FactoryTests, ShouldReturnEnumFromString) {
    // arrange

    ablate::parser::Registrar<FactoryMockClass1>::Register<FactoryMockClass1>(true, "FactoryMockClass1", "this is a simple mock class");
    auto mockFactory = std::make_shared<MockFactory>();
    EXPECT_CALL(*mockFactory, Get(ArgumentIdentifier<std::string>{.inputName = "input123"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("component"));

    // act
    auto argument = ArgumentIdentifier<EnumWrapper<TestEnum>>{.inputName = "input123", .optional = false};
    auto result = std::dynamic_pointer_cast<Factory>(mockFactory)->Get(argument);

    // assert
    ASSERT_EQ(result, TestEnum::COMPONENT);
}

TEST(FactoryTests, ShouldGetMapOfSharedPointers) {
    // arrange
    const std::string defaultClassType = "FactoryMockClass1";
    ablate::parser::Registrar<FactoryMockClass1>::Register<FactoryMockClass1>(true, std::string(defaultClassType), "this is a simple mock class");
    auto mockFactory = std::make_shared<MockFactory>();

    // subChildFactory
    auto subChildFactory = std::make_shared<MockFactory>();
    EXPECT_CALL(*mockFactory, GetFactory("input123")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(subChildFactory));
    EXPECT_CALL(*subChildFactory, GetKeys()).Times(::testing::Exactly(1)).WillOnce(::testing::Return(std::unordered_set<std::string>{"key1", "key2"}));

    auto factoryChild1 = std::make_shared<MockFactory>();
    EXPECT_CALL(*subChildFactory, GetFactory("key1")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(factoryChild1));
    EXPECT_CALL(*factoryChild1, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(defaultClassType));

    auto factoryChild2 = std::make_shared<MockFactory>();
    EXPECT_CALL(*subChildFactory, GetFactory("key2")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(factoryChild2));
    EXPECT_CALL(*factoryChild2, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(defaultClassType));

    // act
    auto argument = ArgumentIdentifier<std::map<std::string, FactoryMockClass1>>{.inputName = "input123", .optional = false};
    auto result = std::dynamic_pointer_cast<Factory>(mockFactory)->Get(argument);

    // assert
    ASSERT_EQ(result.size(), 2);
    ASSERT_TRUE(std::dynamic_pointer_cast<FactoryMockClass1>(result["key1"]) != nullptr);
    ASSERT_TRUE(std::dynamic_pointer_cast<FactoryMockClass1>(result["key2"]) != nullptr);
}
}  // namespace ablateTesting::parser