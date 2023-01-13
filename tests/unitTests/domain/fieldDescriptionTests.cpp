#include <domain/fieldDescription.hpp>
#include <memory>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mockFactory.hpp"
#include "registrar.hpp"

namespace ablateTesting::flow {

using namespace cppParser;

TEST(FieldDescriptionTests, ShouldBeCreatedByFactoryFunction) {
    // arrange
    auto mockFactory = std::make_shared<cppParserTesting::MockFactory>();

    // return a subMockFactory the components in it
    auto mockSubFactory = std::make_shared<cppParserTesting::MockFactory>();
    std::string className = "ablate::domain::FieldDescription";
    EXPECT_CALL(*mockSubFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(className));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::string>{.inputName = "name"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("name1"));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::string>{.inputName = "prefix", .optional = true})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("prefix1"));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::vector<std::string>>{.inputName = "components", .optional = true}))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(std::vector<std::string>{"one", "two", "three"}));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::string>{.inputName = "location", .optional = true})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("AUX"));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::string>{.inputName = "type"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("FE"));
    EXPECT_CALL(*mockSubFactory, Contains(std::string("region"))).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));
    EXPECT_CALL(*mockSubFactory, GetFactory("region")).Times(::testing::Exactly(0));
    EXPECT_CALL(*mockSubFactory, Contains(std::string("options"))).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));
    EXPECT_CALL(*mockSubFactory, GetFactory("options")).Times(::testing::Exactly(0));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::vector<std::string>>{.inputName = "tags", .optional = true}))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(std::vector<std::string>{"tagA", "tagB", "tagC"}));
    EXPECT_CALL(*mockFactory, GetFactory(std::string("input123"))).Times(::testing::Exactly(1)).WillOnce(::testing::Return(mockSubFactory));

    // act
    auto argument = ArgumentIdentifier<ablate::domain::FieldDescription>{.inputName = "input123"};
    auto flowField = std::dynamic_pointer_cast<Factory>(mockFactory)->Get(argument);

    // assert
    ASSERT_EQ(flowField->location, ablate::domain::FieldLocation::AUX);
    ASSERT_EQ(flowField->name, "name1");
    ASSERT_EQ(flowField->prefix, "prefix1_");
    ASSERT_EQ(flowField->components.size(), 3);
    auto expectedComponentNames = std::vector<std::string>{"one", "two", "three"};
    ASSERT_EQ(flowField->components, expectedComponentNames);
    ASSERT_EQ(flowField->type, ablate::domain::FieldType::FEM);
    ASSERT_EQ(flowField->region, nullptr);
    std::vector<std::string> expectedTags{"tagA", "tagB", "tagC"};
    ASSERT_EQ(flowField->tags, expectedTags);
}

TEST(FieldDescriptionTests, ShouldBeCreatedByFactoryFunctionWithMinimalInputs) {
    // arrange
    auto mockFactory = std::make_shared<cppParserTesting::MockFactory>();

    // return a subMockFactory the components in it
    auto mockSubFactory = std::make_shared<cppParserTesting::MockFactory>();
    std::string className = "ablate::domain::FieldDescription";
    EXPECT_CALL(*mockSubFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(className));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::string>{.inputName = "name"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("name1"));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::string>{.inputName = "prefix", .optional = true})).Times(::testing::Exactly(1)).WillOnce(::testing::Return(std::string()));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::vector<std::string>>{.inputName = "components", .optional = true}))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(std::vector<std::string>{}));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::string>{.inputName = "location", .optional = true})).Times(::testing::Exactly(1)).WillOnce(::testing::Return(""));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::string>{.inputName = "type"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("FE"));
    EXPECT_CALL(*mockSubFactory, Contains(std::string("region"))).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));
    EXPECT_CALL(*mockSubFactory, GetFactory("region")).Times(::testing::Exactly(0));
    EXPECT_CALL(*mockSubFactory, Contains(std::string("options"))).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));
    EXPECT_CALL(*mockSubFactory, GetFactory("options")).Times(::testing::Exactly(0));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::vector<std::string>>{.inputName = "tags", .optional = true}))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(std::vector<std::string>{}));
    EXPECT_CALL(*mockFactory, GetFactory(std::string("input123"))).Times(::testing::Exactly(1)).WillOnce(::testing::Return(mockSubFactory));

    // act
    auto argument = ArgumentIdentifier<ablate::domain::FieldDescription>{.inputName = "input123"};
    auto flowField = std::dynamic_pointer_cast<Factory>(mockFactory)->Get(argument);

    // assert
    ASSERT_EQ(flowField->location, ablate::domain::FieldLocation::SOL);
    ASSERT_EQ(flowField->name, "name1");
    ASSERT_EQ(flowField->prefix, "name1_");
    ASSERT_EQ(flowField->components.size(), 1);
    auto expectedComponentNames = std::vector<std::string>{"_"};
    ASSERT_EQ(flowField->components, expectedComponentNames);
    ASSERT_EQ(flowField->type, ablate::domain::FieldType::FEM);
    ASSERT_EQ(flowField->region, nullptr);
}

TEST(FieldDescriptionTests, ShouldActAsSingleFieldDescriptor) {
    // arrange
    auto mockFactory = std::make_shared<cppParserTesting::MockFactory>();

    // return a subMockFactory the components in it
    auto mockSubFactory = std::make_shared<cppParserTesting::MockFactory>();
    std::string className = "ablate::domain::FieldDescription";
    EXPECT_CALL(*mockSubFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(className));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::string>{.inputName = "name"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("name1"));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::string>{.inputName = "prefix", .optional = true})).Times(::testing::Exactly(1)).WillOnce(::testing::Return(std::string()));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::vector<std::string>>{.inputName = "components", .optional = true}))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(std::vector<std::string>{}));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::string>{.inputName = "location", .optional = true})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("AUX"));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::string>{.inputName = "type"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("FE"));
    EXPECT_CALL(*mockSubFactory, Contains(std::string("region"))).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));
    EXPECT_CALL(*mockSubFactory, GetFactory("region")).Times(::testing::Exactly(0));
    EXPECT_CALL(*mockSubFactory, Contains(std::string("options"))).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));
    EXPECT_CALL(*mockSubFactory, GetFactory("options")).Times(::testing::Exactly(0));
    EXPECT_CALL(*mockSubFactory, Get(ArgumentIdentifier<std::vector<std::string>>{.inputName = "tags", .optional = true}))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(std::vector<std::string>{}));
    EXPECT_CALL(*mockFactory, GetFactory(std::string("input123"))).Times(::testing::Exactly(1)).WillOnce(::testing::Return(mockSubFactory));

    // act
    auto argument = ArgumentIdentifier<ablate::domain::FieldDescriptor>{.inputName = "input123"};
    auto fieldDescriptor = std::dynamic_pointer_cast<Factory>(mockFactory)->Get(argument);
    auto fieldDescription = fieldDescriptor->GetFields();

    // assert
    ASSERT_EQ(fieldDescription.size(), 1);
    ASSERT_EQ(fieldDescription.front()->location, ablate::domain::FieldLocation::AUX);
    ASSERT_EQ(fieldDescription.front()->name, "name1");
    ASSERT_EQ(fieldDescription.front()->prefix, "name1_");
    ASSERT_EQ(fieldDescription.front()->components.size(), 1);
    auto expectedComponentNames = std::vector<std::string>{"_"};
    ASSERT_EQ(fieldDescription.front()->components, expectedComponentNames);
    ASSERT_EQ(fieldDescription.front()->type, ablate::domain::FieldType::FEM);
    ASSERT_EQ(fieldDescription.front()->region, nullptr);
}
}  // namespace ablateTesting::flow