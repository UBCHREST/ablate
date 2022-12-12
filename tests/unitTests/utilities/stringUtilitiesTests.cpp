#include <utility>
#include "gtest/gtest.h"
#include "utilities/stringUtilities.hpp"

class StringToUpperTestFixture : public ::testing::TestWithParam<std::pair<std::string, std::string>> {};

TEST_P(StringToUpperTestFixture, ShouldConvertToUpperCase) {
    // arrange
    std::string inputString = GetParam().first;
    // act
    ablate::utilities::StringUtilities::ToUpper(inputString);

    // assert
    ASSERT_EQ(GetParam().second, inputString);
}

TEST_P(StringToUpperTestFixture, ShouldConvertCopyToUpperCase) {
    // arrange
    const std::string inputString = GetParam().first;
    // act
    auto copy = ablate::utilities::StringUtilities::ToUpperCopy(inputString);

    // assert
    ASSERT_EQ(GetParam().second, copy);
}

INSTANTIATE_TEST_SUITE_P(StringUtilititiesTests, StringToUpperTestFixture,
                         testing::Values(std::make_pair("double", "DOUBLE"), std::make_pair("a eAd Ewer", "A EAD EWER"), std::make_pair("  \tCaT BLUE\tGreen ", "  \tCAT BLUE\tGREEN ")));

class StringToLowerTestFixture : public ::testing::TestWithParam<std::pair<std::string, std::string>> {};

TEST_P(StringToLowerTestFixture, ShouldConvertToLowerCase) {
    // arrange
    std::string inputString = GetParam().first;
    // act
    ablate::utilities::StringUtilities::ToLower(inputString);

    // assert
    ASSERT_EQ(GetParam().second, inputString);
}

TEST_P(StringToLowerTestFixture, ShouldConvertCopyToLowerCase) {
    // arrange
    const std::string inputString = GetParam().first;
    // act
    auto copy = ablate::utilities::StringUtilities::ToLowerCopy(inputString);

    // assert
    ASSERT_EQ(GetParam().second, copy);
}

INSTANTIATE_TEST_SUITE_P(StringUtilititiesTests, StringToLowerTestFixture,
                         testing::Values(std::make_pair("DOUBLE", "double"), std::make_pair("a eAd Ewer", "a ead ewer"), std::make_pair("  \tCaT BLUE\tGreen ", "  \tcat blue\tgreen ")));

class StringPrefixTestFixture : public ::testing::TestWithParam<std::tuple<std::string, std::string, bool>> {};

TEST_P(StringPrefixTestFixture, ShouldDetermineStartsWith) {
    // arrange
    std::string inputString = std::get<0>(GetParam());
    std::string inputPrefix = std::get<1>(GetParam());

    // act
    auto startsWith = ablate::utilities::StringUtilities::StartsWith(inputString, inputPrefix);

    // assert
    ASSERT_EQ(startsWith, std::get<2>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(StringUtilititiesTests, StringPrefixTestFixture,
                         testing::Values(std::make_tuple("YiCO2", "Yi", true), std::make_tuple("YICO2", "Yi", false), std::make_tuple("CO2Yi", "Yi", false), std::make_tuple("", "Yi", false),
                                         std::make_tuple("Yi CO2", "Yi", true), std::make_tuple(" Yi CO2", "Yi", false), std::make_tuple(" Yi CO2", " Yi", true)));

class StringPostfixTestFixture : public ::testing::TestWithParam<std::tuple<std::string, std::string, bool>> {};

TEST_P(StringPostfixTestFixture, ShouldDetermineEndsWith) {
    // arrange
    std::string inputString = std::get<0>(GetParam());
    std::string inputPrefix = std::get<1>(GetParam());

    // act
    auto endsWith = ablate::utilities::StringUtilities::EndsWith(inputString, inputPrefix);

    // assert
    ASSERT_EQ(endsWith, std::get<2>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(StringUtilititiesTests, StringPostfixTestFixture,
                         testing::Values(std::make_tuple("CO2Yi", "Yi", true), std::make_tuple("CO2YI", "Yi", false), std::make_tuple("YiCO2", "Yi", false), std::make_tuple("", "Yi", false),
                                         std::make_tuple("CO2 Yi", "Yi", true), std::make_tuple("CO2Yi ", "Yi", false), std::make_tuple("CO2 Yi ", "Yi ", true)));
