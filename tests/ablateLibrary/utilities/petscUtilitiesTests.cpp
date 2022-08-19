#include <utility>
#include "gtest/gtest.h"
#include "utilities/petscUtilities.hpp"
#include "utilities/stringUtilities.hpp"

class PetscDataTypesToStringTestFixture : public ::testing::TestWithParam<std::pair<std::string, PetscDataType>> {};

TEST_P(PetscDataTypesToStringTestFixture, ShouldConvertFromPetscDataTypeToString) {
    // arrange
    std::stringstream stream;
    // act
    using namespace ablate::utilities;
    stream << GetParam().second;
    // assert
    ASSERT_EQ(ablate::utilities::StringUtilities::ToLowerCopy(GetParam().first), ablate::utilities::StringUtilities::ToLowerCopy(stream.str()));
}

INSTANTIATE_TEST_SUITE_P(PetscUtilititiesTests, PetscDataTypesToStringTestFixture,
                         testing::Values(std::make_pair("double", PETSC_DOUBLE), std::make_pair("long", PETSC_LONG), std::make_pair("complex", PETSC_COMPLEX)));

class PetscDataTypesFromStringTestFixture : public ::testing::TestWithParam<std::pair<std::string, PetscDataType>> {};

TEST_P(PetscDataTypesFromStringTestFixture, ShouldConvertFromStringToPetscDataType) {
    // arrange
    std::stringstream stream(GetParam().first);
    // act
    PetscDataType value;
    using namespace ablate::utilities;
    stream >> value;

    // assert
    ASSERT_EQ(GetParam().second, value);
}

INSTANTIATE_TEST_SUITE_P(PetscUtilititiesTests, PetscDataTypesFromStringTestFixture,
                         testing::Values(std::make_pair("double", PETSC_REAL), std::make_pair("real", PETSC_REAL), std::make_pair("scalar", PETSC_SCALAR), std::make_pair("real", PETSC_REAL),
                                         std::make_pair("long", PETSC_LONG), std::make_pair("complex", PETSC_COMPLEX)));
