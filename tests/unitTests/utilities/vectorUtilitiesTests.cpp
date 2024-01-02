#include "gtest/gtest.h"
#include "utilities/vectorUtilities.hpp"

TEST(VectorUtilitiesTests, ShouldConvertVectorsToArraysWithDefaults) {
    {
        auto actual = ablate::utilities::VectorUtilities::ToArray<double, 3>({1.0, 2.0, 3.0}, 0.0);
        auto expected = std::array<double, 3>{1.0, 2.0, 3.0};
        ASSERT_EQ(expected, actual);
    }
    {
        auto actual = ablate::utilities::VectorUtilities::ToArray<double, 3>({1.0, 2.0}, 0.0);
        auto expected = std::array<double, 3>{1.0, 2.0, 0.0};
        ASSERT_EQ(expected, actual);
    }
    {
        auto actual = ablate::utilities::VectorUtilities::ToArray<double, 3>({1.0, 2.0}, 100.0);
        auto expected = std::array<double, 3>{1.0, 2.0, 100.0};
        ASSERT_EQ(expected, actual);
    }
}

TEST(VectorUtilitiesTests, ShouldConvertVectorsToArrays) {
    {
        auto actual = ablate::utilities::VectorUtilities::ToArray<double, 3>({1.0, 2.0, 3.0});
        auto expected = std::array<double, 3>{1.0, 2.0, 3.0};
        ASSERT_EQ(expected, actual);
    }
    {
        auto actual = ablate::utilities::VectorUtilities::ToArray<double, 2>({1.0, 2.0});
        auto expected = std::array<double, 2>{1.0, 2.0};
        ASSERT_EQ(expected, actual);
    }
}

TEST(VectorUtilitiesTests, ShouldConvertThrowExceptionWhenTooLarge) { ASSERT_ANY_THROW((ablate::utilities::VectorUtilities::ToArray<double, 3>({1.0, 2.0, 3.0, 4.0}, 0.0))); }

TEST(VectorUtilitiesTests, ShouldConvertThrowExceptionWhenWrongSize) {
    ASSERT_ANY_THROW((ablate::utilities::VectorUtilities::ToArray<double, 3>({1.0, 2.0, 3.0, 4.0}))) << "When the vector to too large";
    ASSERT_ANY_THROW((ablate::utilities::VectorUtilities::ToArray<double, 5>({1.0, 2.0, 3.0, 4.0}))) << "When the vector to too small";
}