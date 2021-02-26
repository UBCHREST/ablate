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
    MockClass1(std::shared_ptr<Factory> factory){};
};

TEST(RegistrarTests, ShouldRegisterClassAndRecordInLog) {
    // arrange
    auto mockListing = std::make_shared<MockListing>();
    EXPECT_CALL(*mockListing, RecordListing(Listing::ClassEntry{.interface = typeid(MockInterface).name(), .className = "mockClass1", .description = "this is a simple mock class"}))
        .Times(::testing::Exactly(1));

    Listing::ReplaceListing(mockListing);

    // act
    Registrar<MockInterface>::Register<MockClass1>(false, "mockClass1", "this is a simple mock class");

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
                                          .className = "MockClass2",
                                          .description = "this is a simple mock class",
                                          .arguments = {Listing::ArgumentEntry{.name = "dog", .interface = typeid(std::string).name(), .description = "this is a string"},
                                                        Listing::ArgumentEntry{.name = "cat", .interface = typeid(int).name(), .description = "this is a int"},
                                                        Listing::ArgumentEntry{.name = "bird", .interface = typeid(MockInterface).name(), .description = "this is a shared pointer to an interface"}}}))
        .Times(::testing::Exactly(1));

    Listing::ReplaceListing(mockListing);

    // act
    ablate::parser::Registrar<MockInterface>::Register<MockClass2>(false,
                                                                   "MockClass2",
                                                                   "this is a simple mock class",
                                                                   ablate::parser::ArgumentIdentifier<std::string>{"dog", "this is a string"},
                                                                   ablate::parser::ArgumentIdentifier<int>{"cat", "this is a int"},
                                                                   ablate::parser::ArgumentIdentifier<MockInterface>{"bird", "this is a shared pointer to an interface"});

    // assert
    auto createMethod = Registrar<MockInterface>::GetCreateMethod("MockClass2");
    ASSERT_TRUE(createMethod != nullptr);

    // cleanup
    Listing::ReplaceListing(nullptr);
}

class MockInterface4 {
    virtual void Test(){};
};

class MockClass4 : public MockInterface4 {
   public:
    MockClass4(std::string, int, std::shared_ptr<MockInterface4>){};
};

TEST(RegistrarTests, ShouldRegisterDefaultClassWithArgumentIdentifiersAndRecordInLog) {
    // arrange
    auto mockListing = std::make_shared<MockListing>();
    EXPECT_CALL(
        *mockListing,
        RecordListing(Listing::ClassEntry{.interface = typeid(MockInterface4).name(),
                                          .className = "MockClass4",
                                          .description = "this is a simple mock class",
                                          .arguments = {Listing::ArgumentEntry{.name = "dog", .interface = typeid(std::string).name(), .description = "this is a string"},
                                                        Listing::ArgumentEntry{.name = "cat", .interface = typeid(int).name(), .description = "this is a int"},
                                                        Listing::ArgumentEntry{.name = "bird", .interface = typeid(MockInterface4).name(), .description = "this is a shared pointer to an interface"}},
                                          .defaultConstructor = true}))
        .Times(::testing::Exactly(1));

    Listing::ReplaceListing(mockListing);

    // act
    ablate::parser::Registrar<MockInterface4>::Register<MockClass4>(true,
                                                                    "MockClass4",
                                                                    "this is a simple mock class",
                                                                    ablate::parser::ArgumentIdentifier<std::string>{"dog", "this is a string"},
                                                                    ablate::parser::ArgumentIdentifier<int>{"cat", "this is a int"},
                                                                    ablate::parser::ArgumentIdentifier<MockInterface4>{"bird", "this is a shared pointer to an interface"});

    // assert
    auto createMethod = Registrar<MockInterface4>::GetCreateMethod("MockClass4");
    ASSERT_TRUE(createMethod != nullptr);

    // cleanup
    Listing::ReplaceListing(nullptr);
}

class MockInterface5 {
    virtual void Test(){};
};

class MockClass5a : public MockInterface5 {
   public:
    MockClass5a(std::string, int, std::shared_ptr<MockInterface5>){};
};

class MockClass5b : public MockInterface5 {
   public:
    MockClass5b(std::string, int, std::shared_ptr<MockInterface5>){};
};

TEST(RegistrarTests, ShouldNotAllowDoubleDefaultRegistar) {
    // arrange
    ablate::parser::Registrar<MockInterface5>::Register<MockClass5a>(true,
                                                                     "MockClass5a",
                                                                     "this is a simple mock class",
                                                                     ablate::parser::ArgumentIdentifier<std::string>{"dog", "this is a string"},
                                                                     ablate::parser::ArgumentIdentifier<int>{"cat", "this is a int"},
                                                                     ablate::parser::ArgumentIdentifier<MockInterface5>{"bird", "this is a shared pointer to an interface"});

    // act
    // assert
    ASSERT_THROW(ablate::parser::Registrar<MockInterface5>::Register<MockClass5b>(true,
                                                                                  "MockClass5b",
                                                                                  "this is a simple mock class",
                                                                                  ablate::parser::ArgumentIdentifier<std::string>{"dog", "this is a string"},
                                                                                  ablate::parser::ArgumentIdentifier<int>{"cat", "this is a int"},
                                                                                  ablate::parser::ArgumentIdentifier<MockInterface5>{"bird", "this is a shared pointer to an interface"}),
                 std::invalid_argument);
}

TEST(RegistrarTests, ShouldResolveAndCreate) {
    // arrange
    auto mockFactory = std::make_shared<MockFactory>();
    const std::string expectedClassType = "mockClass1";

    EXPECT_CALL(*mockFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedClassType));

    Registrar<MockInterface>::Register<MockClass1>(false, "mockClass1", "this is a simple mock class");

    // act
    auto instance = ResolveAndCreate<MockInterface>(mockFactory);

    // assert
    ASSERT_TRUE(instance != nullptr) << " should create an instance of the interface";
    ASSERT_TRUE(std::dynamic_pointer_cast<MockClass1>(instance) != nullptr) << " should be an instance of MockClass1";
}

TEST(RegistrarTests, ShouldThrowExceptionWhenCannotResolveAndCreate) {
    // arrange
    auto mockFactory = std::make_shared<MockFactory>();
    const std::string expectedClassType = "mockClass34";

    EXPECT_CALL(*mockFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedClassType));

    Registrar<MockInterface>::Register<MockClass1>(false, "mockClass1", "this is a simple mock class");

    // act
    // assert
    ASSERT_THROW(ResolveAndCreate<MockInterface>(mockFactory), std::invalid_argument);
}

TEST(RegistrarTests, ShouldCreateDefaultAndUseWhenNotSpecified) {
    // arrange
    auto mockFactory = std::make_shared<MockFactory>();
    const std::string expectedClassType = "";

    EXPECT_CALL(*mockFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedClassType));

    Registrar<MockInterface>::Register<MockClass1>(true, "mockClass54", "this is a simple mock class");

    // act
    auto result = ResolveAndCreate<MockInterface>(mockFactory);

    // assert
    ASSERT_TRUE(result != nullptr);
}

class NoDefaultInterface {
    virtual void Test(){};
};

class MockClass3 : public NoDefaultInterface {
   public:
    MockClass3(std::shared_ptr<Factory> factory){};
};

TEST(RegistrarTests, ShouldThrowExceptionWhenNoDefaultIsSpecified) {
    // arrange
    auto mockFactory = std::make_shared<MockFactory>();
    const std::string expectedClassType = "";

    EXPECT_CALL(*mockFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedClassType));

    Registrar<NoDefaultInterface>::Register<MockClass3>(false, "mockClass2", "this is a simple mock class");

    // act
    // assert
    ASSERT_THROW(ResolveAndCreate<NoDefaultInterface>(mockFactory), std::invalid_argument);
}
}  // namespace ablateTesting::parser