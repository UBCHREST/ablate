#include <flow/flowFieldDescriptor.hpp>
#include <memory>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "parser/mockFactory.hpp"
#include "parser/registrar.hpp"

namespace ablateTesting::flow {

using namespace ablate::parser;

TEST(FlowFieldDescriptors, ShouldBeCreatedByFactoryFunction) {
    // arrange
    auto mockFactory = std::make_shared<ablateTesting::parser::MockFactory>();

    // return a subMockFactory the the components in it
    auto mockSubFactory = std::make_shared<ablateTesting::parser::MockFactory>();
    std::string className = ""; /* use default value*/
    EXPECT_CALL(*mockSubFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(className));
    EXPECT_CALL(*mockSubFactory, Contains(std::string("solutionField"))).Times(::testing::Exactly(1)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<bool>{.inputName = "solutionField"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::string>{.inputName = "fieldName"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("name1"));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::string>{.inputName = "fieldPrefix"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("prefix1"));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<int>{.inputName = "components"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return(3));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::string>{.inputName = "fieldType"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("FV"));
    EXPECT_CALL(*mockSubFactory, Contains(std::string("componentNames"))).Times(::testing::Exactly(1)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::vector<std::string>>{.inputName = "componentNames"}))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(std::vector<std::string>{"one", "two", "three"}));

    EXPECT_CALL(*mockFactory, GetFactory(std::string("input123"))).Times(::testing::Exactly(1)).WillOnce(::testing::Return(mockSubFactory));

    // act
    auto argument = ArgumentIdentifier<ablate::flow::FlowFieldDescriptor>{.inputName = "input123"};
    auto flowField = std::dynamic_pointer_cast<Factory>(mockFactory)->Get(argument);

    // assert
    ASSERT_EQ(flowField->solutionField, false);
    ASSERT_EQ(flowField->fieldName, "name1");
    ASSERT_EQ(flowField->fieldPrefix, "prefix1");
    ASSERT_EQ(flowField->components, 3);
    ASSERT_EQ(flowField->fieldType, ablate::flow::FieldType::FV);
    auto expectedComponentNames = std::vector<std::string>{"one", "two", "three"};
    ASSERT_EQ(flowField->componentNames, expectedComponentNames);
}
}  // namespace ablateTesting::flow