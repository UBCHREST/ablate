#include <memory>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mockFactory.hpp"
#include "parser/registrar.hpp"

using ::testing::AtLeast;

namespace ablateTesting::parser {

using namespace ablate::parser;

class MockInterface {
    virtual void Test(){};
};

class MockListing : public ablate::parser::Listing {
   public:
    MOCK_METHOD(void, RecordListing, (ClassEntry entry));
};

class MockClass1 : public MockInterface {
   public:
    MockClass1(Factory& factory){};
};

TEST(RegistrarTests, ShouldRegisterClassAndRecordInLog) {
    // arrange
    auto mockListing = std::make_shared<MockListing>();
    EXPECT_CALL(*mockListing, RecordListing(Listing::ClassEntry{.interface = typeid(MockInterface).name(), .description = "this is a simple mock class", .className = "mockClass1"}))
        .Times(::testing::Exactly(1));

    Listing::ReplaceListing(mockListing);

    // act
    Registrar<MockInterface>::Register<MockClass1>("mockClass1", "this is a simple mock class");

    // assert
    auto createMethod = Registrar<MockInterface>::GetCreateMethod("mockClass1");
    ASSERT_TRUE(createMethod != nullptr);

    // cleanup
    Listing::ReplaceListing(nullptr);
}

class MockClass2 : public MockInterface {
   public:
    MockClass2(std::string, int, std::shared_ptr<MockInterface>){};
};

TEST(RegistrarTests, ShouldRegisterClassWithArgumentIdentifiersAndRecordInLog) {
    // arrange
    auto mockListing = std::make_shared<MockListing>();
    EXPECT_CALL(
        *mockListing,
        RecordListing(Listing::ClassEntry{.interface = typeid(MockInterface).name(),
                                          .description = "this is a simple mock class",
                                          .className = "MockClass2",
                                          .arguments = {Listing::ArgumentEntry{.name = "dog", .interface = typeid(std::string).name(), .description = "this is a string"},
                                                        Listing::ArgumentEntry{.name = "cat", .interface = typeid(int).name(), .description = "this is a int"},
                                                        Listing::ArgumentEntry{.name = "bird", .interface = typeid(MockInterface).name(), .description = "this is a shared pointer to an interface"}}}))
        .Times(::testing::Exactly(1));

    Listing::ReplaceListing(mockListing);

    // act
    REGISTER(MockInterface,MockClass2, "this is a simple mock class",
             ARG(std::string, "dog", "this is a string"),
             ARG(int, "cat", "this is a int"),
             ARG(MockInterface, "bird", "this is a shared pointer to an interface")

    );

    // assert
    auto createMethod = Registrar<MockInterface>::GetCreateMethod("MockClass2");
    ASSERT_TRUE(createMethod != nullptr);

    // cleanup
    Listing::ReplaceListing(nullptr);
}

TEST(RegistrarTests, ShouldResolveAndCreate) {
    // arrange
    auto mockFactory = MockFactory();
    const std::string expectedClassType = "mockClass1";

    EXPECT_CALL(mockFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedClassType));

    Registrar<MockInterface>::Register<MockClass1>("mockClass1", "this is a simple mock class");

    // act
    auto instance = ResolveAndCreate<MockInterface>(mockFactory);

    // assert
    ASSERT_TRUE(instance != nullptr) << " should create an instance of the interface";
    ASSERT_TRUE(std::dynamic_pointer_cast<MockClass1>(instance) != nullptr) << " should be an instance of MockClass1";
}

//TEST(RegistrarTests, ShouldThrowExceptionWhenCannotResolveAndCreate) {
//    // arrange
//    auto mockFactory = MockFactory();
//    const std::string expectedClassType = "mockClass34";
//
//    EXPECT_CALL(mockFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedClassType));
//
//    Registrar<MockInterface>::Register<MockClass1>("mockClass1", "this is a simple mock class");
//
//    // act
//    // assert
//    ASSERT_THROW(ResolveAndCreate<MockInterface>(mockFactory), std::invalid_argument);
//}
}