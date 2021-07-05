#include <memory>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mockFactory.hpp"
#include "parser/registrar.hpp"

using ::testing::AtLeast;

namespace ablateTesting::parser {

using namespace ablate::parser;

class MockInterface {
   public:
    virtual ~MockInterface() = default;
    virtual void Test(){};
};

class MockListing : public ablate::parser::Listing {
   public:
    ~MockListing() override = default;
    MOCK_METHOD(void, RecordListing, (ClassEntry entry));
};

class MockClass1 : public MockInterface {
   public:
    ~MockClass1() override = default;
    MockClass1(std::shared_ptr<Factory> factory){};
};

TEST(RegistrarTests, ShouldRegisterClassAndRecordInLog) {
    // arrange
    auto mockListing = std::make_shared<MockListing>();
    EXPECT_CALL(*mockListing, RecordListing(Listing::ClassEntry{.interface = typeid(MockInterface).name(), .className = "mockClass1", .description = "this is a simple mock class"}))
        .Times(::testing::Exactly(1));

    Listing::ReplaceListing(mockListing);

    // act
    Registrar<MockInterface>::RegisterWithFactoryConstructor<MockClass1>(false, "mockClass1", "this is a simple mock class");

    // assert
    auto createMethod = Registrar<MockInterface>::GetCreateMethod("mockClass1");
    ASSERT_TRUE(createMethod != nullptr);

    // cleanup
    Listing::ReplaceListing(nullptr);
}

class MockClass2 : public MockInterface {
   public:
    ~MockClass2() override = default;
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

TEST(RegistrarTests, ShouldRegisterClassWithArgumentIdentifiersAndOptAndRecordInLog) {
    // arrange
    auto mockListing = std::make_shared<MockListing>();
    EXPECT_CALL(*mockListing,
                RecordListing(Listing::ClassEntry{
                    .interface = typeid(MockInterface).name(),
                    .className = "MockClass2a",
                    .description = "this is a simple mock class",
                    .arguments = {Listing::ArgumentEntry{.name = "dog", .interface = typeid(std::string).name(), .description = "this is a string", .optional = true},
                                  Listing::ArgumentEntry{.name = "cat", .interface = typeid(int).name(), .description = "this is a int", .optional = true},
                                  Listing::ArgumentEntry{.name = "bird", .interface = typeid(MockInterface).name(), .description = "this is a shared pointer to an interface", .optional = true}}}))
        .Times(::testing::Exactly(1));

    Listing::ReplaceListing(mockListing);

    // act
    ablate::parser::Registrar<MockInterface>::Register<MockClass2>(false,
                                                                   "MockClass2a",
                                                                   "this is a simple mock class",
                                                                   ablate::parser::ArgumentIdentifier<std::string>{"dog", "this is a string", true},
                                                                   ablate::parser::ArgumentIdentifier<int>{"cat", "this is a int", true},
                                                                   ablate::parser::ArgumentIdentifier<MockInterface>{"bird", "this is a shared pointer to an interface", true});

    // assert
    auto createMethod = Registrar<MockInterface>::GetCreateMethod("MockClass2a");
    ASSERT_TRUE(createMethod != nullptr);

    // cleanup
    Listing::ReplaceListing(nullptr);
}

class MockInterface4 {
   public:
    virtual ~MockInterface4() = default;
    virtual void Test(){};
};

class MockClass4 : public MockInterface4 {
   public:
    ~MockClass4() override = default;
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
   public:
    virtual ~MockInterface5() = default;
    virtual void Test(){};
};

class MockClass5a : public MockInterface5 {
   public:
    ~MockClass5a() override = default;
    MockClass5a(std::string, int, std::shared_ptr<MockInterface5>){};
};

class MockClass5b : public MockInterface5 {
   public:
    ~MockClass5b() override = default;
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

    Registrar<MockInterface>::RegisterWithFactoryConstructor<MockClass1>(false, "mockClass1", "this is a simple mock class");

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

    Registrar<MockInterface>::RegisterWithFactoryConstructor<MockClass1>(false, "mockClass1", "this is a simple mock class");

    // act
    // assert
    ASSERT_THROW(ResolveAndCreate<MockInterface>(mockFactory), std::invalid_argument);
}

TEST(RegistrarTests, ShouldCreateDefaultAndUseWhenNotSpecified) {
    // arrange
    auto mockFactory = std::make_shared<MockFactory>();
    const std::string expectedClassType = "";

    EXPECT_CALL(*mockFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedClassType));

    Registrar<MockInterface>::RegisterWithFactoryConstructor<MockClass1>(true, "mockClass54", "this is a simple mock class");

    // act
    auto result = ResolveAndCreate<MockInterface>(mockFactory);

    // assert
    ASSERT_TRUE(result != nullptr);
}

class NoDefaultInterface {
   public:
    virtual ~NoDefaultInterface() = default;
    virtual void Test(){};
};

class MockClass3 : public NoDefaultInterface {
   public:
    ~MockClass3() override = default;
    MockClass3(std::shared_ptr<Factory> factory){};
};

TEST(RegistrarTests, ShouldThrowExceptionWhenNoDefaultIsSpecified) {
    // arrange
    auto mockFactory = std::make_shared<MockFactory>();
    const std::string expectedClassType = "";

    EXPECT_CALL(*mockFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedClassType));

    Registrar<NoDefaultInterface>::RegisterWithFactoryConstructor<MockClass3>(false, "mockClass2", "this is a simple mock class");

    // act
    // assert
    ASSERT_THROW(ResolveAndCreate<NoDefaultInterface>(mockFactory), std::invalid_argument);
}

class MockClass6 : public MockInterface {
   public:
    const int a;
    const int b;

   public:
    ~MockClass6() override = default;
    MockClass6(int a, int b) : a(a), b(b){};
};

static std::shared_ptr<MockClass6> MakeMockClass6Function(std::shared_ptr<Factory> factory) {
    auto c = factory->GetByName<int>("c");
    return std::make_shared<MockClass6>(c * 2, c * 3);
}

TEST(RegistrarTests, ShouldRegisterFunctionForClassAndRecordInLog) {
    // arrange
    auto mockListing = std::make_shared<MockListing>();
    EXPECT_CALL(*mockListing, RecordListing(Listing::ClassEntry{.interface = typeid(MockInterface).name(), .className = "mockClass6", .description = "this is a simple mock class"}))
        .Times(::testing::Exactly(1));

    Listing::ReplaceListing(mockListing);

    // act
    Registrar<MockInterface>::RegisterWithFactoryFunction<MockClass6>(false, "mockClass6", "this is a simple mock class", MakeMockClass6Function);

    // assert
    auto createMethod = Registrar<MockInterface>::GetCreateMethod("mockClass6");
    ASSERT_TRUE(createMethod != nullptr);

    // cleanup
    Listing::ReplaceListing(nullptr);
}

}  // namespace ablateTesting::parser