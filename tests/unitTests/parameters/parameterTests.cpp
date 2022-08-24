#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "parameters/mockParameters.hpp"
#include "parameters/parameters.hpp"

using ::testing::AtLeast;

namespace ablateTesting::parameters {

using namespace ablate::parameters;

// double based tests
class ParameterTestFixtureDouble : public testing::TestWithParam<std::tuple<std::string, double>> {};

TEST_P(ParameterTestFixtureDouble, GetShouldReturnValue) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(expectedString));

    // act
    auto actualValue = mockParameters.Get<double>(key);

    // assert
    EXPECT_TRUE(actualValue.has_value());
    EXPECT_DOUBLE_EQ(actualValue.value(), expectedValue);
}

TEST_P(ParameterTestFixtureDouble, GetShouldReturnEmptyOptional) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1));

    // act
    auto actualValue = mockParameters.Get<double>(key);

    // assert
    EXPECT_FALSE(actualValue.has_value());
}

TEST_P(ParameterTestFixtureDouble, GetExpectShouldReturnValue) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(expectedString));

    // act
    auto actualValue = mockParameters.GetExpect<double>(key);

    // assert
    EXPECT_DOUBLE_EQ(actualValue, expectedValue);
}

TEST_P(ParameterTestFixtureDouble, GetShouldThrowExceptionWhenNotFound) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1));

    // act
    // assert
    EXPECT_THROW(mockParameters.GetExpect<double>(key), ParameterException);
}

TEST_P(ParameterTestFixtureDouble, GetExpectShouldReturnValueEvenWithDefault) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(expectedString));

    // act
    double actualValue = mockParameters.Get<double>(key, 102.2);

    // assert
    EXPECT_DOUBLE_EQ(actualValue, expectedValue);
}

TEST_P(ParameterTestFixtureDouble, GetShouldReturnDefaultValue) {
    // arrange
    const auto [expectedString, _] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1));

    // act
    auto actualValue = mockParameters.Get<double>(key, 102.2);

    // assert
    EXPECT_DOUBLE_EQ(actualValue, 102.2);
}

INSTANTIATE_TEST_SUITE_P(ParameterTests, ParameterTestFixtureDouble, ::testing::Values(std::make_tuple("22.3", 22.3), std::make_tuple(" 1E-3 ", 1.0E-3), std::make_tuple("-1.2", -1.2)));

// int based tests
class ParameterTestFixtureInt : public testing::TestWithParam<std::tuple<std::string, int>> {};

TEST_P(ParameterTestFixtureInt, GetShouldReturnValue) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(expectedString));

    // act
    auto actualValue = mockParameters.Get<int>(key);

    // assert
    EXPECT_TRUE(actualValue.has_value());
    EXPECT_EQ(actualValue.value(), expectedValue);
}

TEST_P(ParameterTestFixtureInt, GetShouldReturnEmptyOptional) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1));

    // act
    auto actualValue = mockParameters.Get<int>(key);

    // assert
    EXPECT_FALSE(actualValue.has_value());
}

TEST_P(ParameterTestFixtureInt, GetExpectShouldReturnValue) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(expectedString));

    // act
    auto actualValue = mockParameters.GetExpect<int>(key);

    // assert
    EXPECT_DOUBLE_EQ(actualValue, expectedValue);
}

TEST_P(ParameterTestFixtureInt, GetShouldThrowExceptionWhenNotFound) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1));

    // act
    // assert
    EXPECT_THROW(mockParameters.GetExpect<int>(key), ParameterException);
}

TEST_P(ParameterTestFixtureInt, GetExpectShouldReturnValueEvenWithDefault) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(expectedString));

    // act
    int actualValue = mockParameters.Get<int>(key, 102);

    // assert
    EXPECT_EQ(actualValue, expectedValue);
}

TEST_P(ParameterTestFixtureInt, GetShouldReturnDefaultValue) {
    // arrange
    const auto [expectedString, _] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1));

    // act
    auto actualValue = mockParameters.Get<int>(key, 102);

    // assert
    EXPECT_EQ(actualValue, 102);
}

INSTANTIATE_TEST_SUITE_P(ParameterTests, ParameterTestFixtureInt, ::testing::Values(std::make_tuple("22.3", 22), std::make_tuple(" 3 ", 3), std::make_tuple("-23", -23)));

// bool based tests
class ParameterTestFixtureBool : public testing::TestWithParam<std::tuple<std::string, bool>> {};

TEST_P(ParameterTestFixtureBool, GetShouldReturnValue) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(expectedString));

    // act
    auto actualValue = mockParameters.Get<bool>(key);

    // assert
    EXPECT_TRUE(actualValue.has_value());
    EXPECT_EQ(actualValue.value(), expectedValue);
}

TEST_P(ParameterTestFixtureBool, GetShouldReturnEmptyOptional) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1));

    // act
    auto actualValue = mockParameters.Get<bool>(key);

    // assert
    EXPECT_FALSE(actualValue.has_value());
}

TEST_P(ParameterTestFixtureBool, GetExpectShouldReturnValue) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(expectedString));

    // act
    auto actualValue = mockParameters.GetExpect<bool>(key);

    // assert
    EXPECT_DOUBLE_EQ(actualValue, expectedValue);
}

TEST_P(ParameterTestFixtureBool, GetShouldThrowExceptionWhenNotFound) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1));

    // act
    // assert
    EXPECT_THROW(mockParameters.GetExpect<bool>(key), ParameterException);
}

TEST_P(ParameterTestFixtureBool, GetExpectShouldReturnValueEvenWithDefault) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(expectedString));

    // act
    int actualValue = mockParameters.Get<bool>(key, true);

    // assert
    EXPECT_EQ(actualValue, expectedValue);
}

TEST_P(ParameterTestFixtureBool, GetShouldReturnDefaultValue) {
    // arrange
    const auto [expectedString, _] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1));

    // act
    auto actualValue = mockParameters.Get<bool>(key, true);

    // assert
    EXPECT_EQ(actualValue, true);
}

INSTANTIATE_TEST_SUITE_P(ParameterTests, ParameterTestFixtureBool,
                         ::testing::Values(std::make_tuple("true", true), std::make_tuple("false", false), std::make_tuple("True", true), std::make_tuple("TRUE", true), std::make_tuple("1", true),
                                           std::make_tuple("0", false), std::make_tuple("yes", true)));

// string based tests
class ParameterTestFixtureString : public testing::TestWithParam<std::tuple<std::string, std::string>> {};

TEST_P(ParameterTestFixtureString, GetShouldReturnValue) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(expectedString));

    // act
    auto actualValue = mockParameters.Get<std::string>(key);

    // assert
    EXPECT_TRUE(actualValue.has_value());
    EXPECT_EQ(actualValue.value(), expectedValue);
}

TEST_P(ParameterTestFixtureString, GetShouldReturnEmptyOptional) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1));

    // act
    auto actualValue = mockParameters.Get<std::string>(key);

    // assert
    EXPECT_FALSE(actualValue.has_value());
}

TEST_P(ParameterTestFixtureString, GetExpectShouldReturnValue) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(expectedString));

    // act
    auto actualValue = mockParameters.GetExpect<std::string>(key);

    // assert
    EXPECT_EQ(actualValue, expectedValue);
}

TEST_P(ParameterTestFixtureString, GetShouldThrowExceptionWhenNotFound) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1));

    // act
    // assert
    EXPECT_THROW(mockParameters.GetExpect<std::string>(key), ParameterException);
}

TEST_P(ParameterTestFixtureString, GetExpectShouldReturnValueEvenWithDefault) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(expectedString));

    // act
    std::string actualValue = mockParameters.Get<std::string>(key, "default_123");

    // assert
    EXPECT_EQ(actualValue, expectedValue);
}

TEST_P(ParameterTestFixtureString, GetShouldReturnDefaultValue) {
    // arrange
    const auto [expectedString, _] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1));

    // act
    auto actualValue = mockParameters.Get<std::string>(key, "default_123");

    // assert
    EXPECT_EQ(actualValue, "default_123");
}

INSTANTIATE_TEST_SUITE_P(ParameterTests, ParameterTestFixtureString, ::testing::Values(std::make_tuple("22.3", "22.3"), std::make_tuple(" 3 ", "3"), std::make_tuple("one_two three ", "one_two")));

// fill tests
TEST(ParameterTestFill, ShouldFillDoubleValues) {
    // arrange
    const char *names[3] = {"strouhal", "reynolds", "peclet"};
    double values[3];

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString("strouhal")).Times(::testing::Exactly(1)).WillOnce(::testing::Return("12.3"));
    EXPECT_CALL(mockParameters, GetString("reynolds")).Times(::testing::Exactly(1)).WillOnce(::testing::Return("4.56"));
    EXPECT_CALL(mockParameters, GetString("peclet")).Times(::testing::Exactly(1)).WillOnce(::testing::Return("7.89"));

    // act
    mockParameters.Fill(3, names, &values[0]);

    // assert
    EXPECT_DOUBLE_EQ(values[0], 12.3);
    EXPECT_DOUBLE_EQ(values[1], 4.56);
    EXPECT_DOUBLE_EQ(values[2], 7.89);
}

TEST(ParameterTestFill, ShouldThrowExceptionForMissingDoubleValues) {
    // arrange
    const char *names[4] = {"strouhal", "reynolds", "other", "peclet"};
    double values[4];

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString("strouhal")).Times(::testing::Exactly(1)).WillOnce(::testing::Return("12.3"));
    EXPECT_CALL(mockParameters, GetString("reynolds")).Times(::testing::Exactly(1)).WillOnce(::testing::Return("4.56"));
    EXPECT_CALL(mockParameters, GetString("other")).Times(::testing::Exactly(1));
    EXPECT_CALL(mockParameters, GetString("peclet")).Times(::testing::Exactly(0));

    // act
    // assert
    EXPECT_THROW(mockParameters.Fill(4, names, &values[0]), ParameterException);
}

TEST(ParameterTestFill, ShouldFillIntValues) {
    // arrange
    const char *names[3] = {"strouhal", "reynolds", "peclet"};
    int values[3];

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString("strouhal")).Times(::testing::Exactly(1)).WillOnce(::testing::Return("12"));
    EXPECT_CALL(mockParameters, GetString("reynolds")).Times(::testing::Exactly(1)).WillOnce(::testing::Return("4"));
    EXPECT_CALL(mockParameters, GetString("peclet")).Times(::testing::Exactly(1)).WillOnce(::testing::Return("7"));

    // act
    mockParameters.Fill(3, names, &values[0]);

    // assert
    EXPECT_EQ(values[0], 12);
    EXPECT_EQ(values[1], 4);
    EXPECT_EQ(values[2], 7);
}

TEST(ParameterTestFill, ShouldThrowExceptionForMissingIntValues) {
    // arrange
    const char *names[4] = {"strouhal", "reynolds", "other", "peclet"};
    int values[4];

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString("strouhal")).Times(::testing::Exactly(1)).WillOnce(::testing::Return("12"));
    EXPECT_CALL(mockParameters, GetString("reynolds")).Times(::testing::Exactly(1)).WillOnce(::testing::Return("4"));
    EXPECT_CALL(mockParameters, GetString("other")).Times(::testing::Exactly(1));
    EXPECT_CALL(mockParameters, GetString("peclet")).Times(::testing::Exactly(0));

    // act
    // assert
    EXPECT_THROW(mockParameters.Fill(4, names, &values[0]), ParameterException);
}

// double vector
class ParameterTestFixtureDoubleVector : public testing::TestWithParam<std::tuple<std::string, std::vector<double>>> {};

TEST_P(ParameterTestFixtureDoubleVector, GetShouldReturnValue) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(expectedString));

    // act
    auto actualValue = mockParameters.Get<std::vector<double>>(key);

    // assert
    EXPECT_TRUE(actualValue.has_value());
    EXPECT_EQ(actualValue.value(), expectedValue);
}

TEST_P(ParameterTestFixtureDoubleVector, GetShouldReturnEmptyOptional) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1));

    // act
    auto actualValue = mockParameters.Get<std::vector<double>>(key);

    // assert
    EXPECT_FALSE(actualValue.has_value());
}

TEST_P(ParameterTestFixtureDoubleVector, GetExpectShouldReturnValue) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(expectedString));

    // act
    auto actualValue = mockParameters.Get<std::vector<double>>(key);

    // assert
    EXPECT_EQ(actualValue, expectedValue);
}

TEST_P(ParameterTestFixtureDoubleVector, GetShouldThrowExceptionWhenNotFound) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1));

    // act
    // assert
    EXPECT_THROW(mockParameters.GetExpect<std::vector<double>>(key), ParameterException);
}

TEST_P(ParameterTestFixtureDoubleVector, GetExpectShouldReturnValueEvenWithDefault) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(expectedString));

    // act
    auto actualValue = mockParameters.Get<std::vector<double>>(key, {102.2});

    // assert
    EXPECT_EQ(actualValue, expectedValue);
}

TEST_P(ParameterTestFixtureDoubleVector, GetShouldReturnDefaultValue) {
    // arrange
    const auto [expectedString, _] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1));

    // act
    auto actualValue = mockParameters.Get<std::vector<double>>(key, {102.2});

    // assert
    EXPECT_EQ(actualValue, std::vector<double>{102.2});
}

INSTANTIATE_TEST_SUITE_P(ParameterTests, ParameterTestFixtureDoubleVector,
                         ::testing::Values(std::make_tuple("22.3", std::vector<double>{22.3}), std::make_tuple("1E-3 2.3 ", std::vector<double>{1.0E-3, 2.3}),
                                           std::make_tuple("", std::vector<double>{})));

// double vector
class ParameterTestFixtureDoubleArray : public testing::TestWithParam<std::tuple<std::string, std::array<double, 4>>> {};

TEST_P(ParameterTestFixtureDoubleArray, GetShouldReturnValue) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(expectedString));

    // act
    auto actualValue = mockParameters.Get<std::array<double, 4>>(key);

    // assert
    EXPECT_TRUE(actualValue.has_value());
    EXPECT_EQ(actualValue.value(), expectedValue);
}

TEST_P(ParameterTestFixtureDoubleArray, GetShouldReturnEmptyOptional) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1));

    // act
    auto actualValue = mockParameters.Get<std::array<double, 4>>(key);

    // assert
    EXPECT_FALSE(actualValue.has_value());
}

TEST_P(ParameterTestFixtureDoubleArray, GetExpectShouldReturnValue) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(expectedString));

    // act
    auto actualValue = mockParameters.Get<std::array<double, 4>>(key);

    // assert
    EXPECT_EQ(actualValue, expectedValue);
}

TEST_P(ParameterTestFixtureDoubleArray, GetShouldThrowExceptionWhenNotFound) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1));

    // act
    // assert
    EXPECT_THROW((mockParameters.GetExpect<std::array<double, 4>>(key)), ParameterException);
}

TEST_P(ParameterTestFixtureDoubleArray, GetExpectShouldReturnValueEvenWithDefault) {
    // arrange
    const auto [expectedString, expectedValue] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(expectedString));

    // act
    auto actualValue = mockParameters.Get<std::array<double, 4>>(key, {101.1, 202.2, 303.3, 404.4});

    // assert
    EXPECT_EQ(actualValue, expectedValue);
}

TEST_P(ParameterTestFixtureDoubleArray, GetShouldReturnDefaultValue) {
    // arrange
    const auto [expectedString, _] = GetParam();
    const std::string key = "key 123";

    MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString(key)).Times(::testing::Exactly(1));

    // act
    auto actualValue = mockParameters.Get<std::array<double, 4>>(key, {101.1, 202.2, 303.3, 404.4});

    // assert
    EXPECT_EQ(actualValue, (std::array<double, 4>{101.1, 202.2, 303.3, 404.4}));
}

INSTANTIATE_TEST_SUITE_P(ParameterTests, ParameterTestFixtureDoubleArray,
                         ::testing::Values(std::make_tuple("22.3", std::array<double, 4>{22.3, 0, 0, 0}), std::make_tuple("1E-3 2.3 ", std::array<double, 4>{1.0E-3, 2.3, 0, 0}),
                                           std::make_tuple("", std::array<double, 4>{0, 0, 0, 0}), std::make_tuple("11.1 22.2 33.3 44.4", std::array<double, 4>{11.1, 22.2, 33.3, 44.4}),
                                           std::make_tuple("11.1 22.2 33.3 44.4 55.5", std::array<double, 4>{11.1, 22.2, 33.3, 44.4})));

}  // namespace ablateTesting::parameters