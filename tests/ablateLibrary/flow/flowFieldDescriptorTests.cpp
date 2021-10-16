#include <memory>
#include <domain/fieldDescriptor.hpp>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "parser/mockFactory.hpp"
#include "parser/registrar.hpp"

namespace ablateTesting::flow {

using namespace ablate::parser;

TEST(FieldDescriptors, ShouldBeCreatedByFactoryFunction) {
    // arrange
    auto mockFactory = std::make_shared<ablateTesting::parser::MockFactory>();

    // return a subMockFactory the the components in it
    auto mockSubFactory = std::make_shared<ablateTesting::parser::MockFactory>();
    std::string className = ""; /* use default value*/
    EXPECT_CALL(*mockSubFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(className));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::string>{.inputName = "name"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("name1"));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::string>{.inputName = "prefix"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("prefix1"));
    EXPECT_CALL(*mockSubFactory, Contains(std::string("components"))).Times(::testing::Exactly(1)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::vector<std::string>>{.inputName = "components"}))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(std::vector<std::string>{"one", "two", "three"}));
    EXPECT_CALL(*mockSubFactory, Contains(std::string("type"))).Times(::testing::Exactly(1)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::string>{.inputName = "type"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("AUX"));

    EXPECT_CALL(*mockFactory, GetFactory(std::string("input123"))).Times(::testing::Exactly(1)).WillOnce(::testing::Return(mockSubFactory));

    // act
    auto argument = ArgumentIdentifier<ablate::domain::FieldDescriptor>{.inputName = "input123"};
    auto flowField = std::dynamic_pointer_cast<Factory>(mockFactory)->Get(argument);

    // assert
    ASSERT_EQ(flowField->type, ablate::domain::FieldType::AUX);
    ASSERT_EQ(flowField->name, "name1");
    ASSERT_EQ(flowField->prefix, "prefix1");
    ASSERT_EQ(flowField->components.size(), 3);
    auto expectedComponentNames = std::vector<std::string>{"one", "two", "three"};
    ASSERT_EQ(flowField->components, expectedComponentNames);
}
}  // namespace ablateTesting::flow