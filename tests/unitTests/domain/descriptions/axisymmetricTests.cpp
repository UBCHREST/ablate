#include <fstream>
#include "domain/descriptions/axisymmetric.hpp"
#include "gtest/gtest.h"

struct AxisymmetricMeshDescriptionParameters {
    // Hold an mesh generator
    std::shared_ptr<ablate::domain::descriptions::Axisymmetric> description;

    // Expected values
    PetscInt expectedMeshDimension;
    PetscInt expectedNumberCells;
    PetscInt expectedNumberVertices;
    std::map<PetscInt, DMPolytopeType> expectedCellTypes;
    std::map<PetscInt, std::vector<PetscInt>> expectedTopology;
};

class AxisymmetricMeshDescriptionFixture : public ::testing::TestWithParam<AxisymmetricMeshDescriptionParameters> {};

TEST_P(AxisymmetricMeshDescriptionFixture, ShouldComputeCorrectNumberOfCellsAndVertices) {
    // arrange
    auto description = GetParam().description;
    // act
    // assert
    ASSERT_EQ(GetParam().expectedNumberCells, description->GetNumberCells());
    ASSERT_EQ(GetParam().expectedNumberVertices, description->GetNumberVertices());
    ASSERT_EQ(GetParam().expectedMeshDimension, description->GetMeshDimension());
}

TEST_P(AxisymmetricMeshDescriptionFixture, ShouldProduceCorrectCellTypes) {
    // arrange
    auto description = GetParam().description;
    // act
    // assert
    for (const auto& test : GetParam().expectedCellTypes) {
        ASSERT_EQ(test.second, description->GetCellType(test.first)) << "For Cell " << test.first << ". Check https://petsc.org/release/manualpages/DM/DMPolytopeType/ for types." << std::endl;
    }
}

TEST_P(AxisymmetricMeshDescriptionFixture, ShouldBuildCorrectTopology) {
    // arrange
    auto description = GetParam().description;
    // act
    // assert
    for (const auto& test : GetParam().expectedTopology) {
        // Build the testCellNodes
        std::vector<PetscInt> testNodes(test.second.size());
        description->BuildTopology(test.first, testNodes.data());

        ASSERT_EQ(test.second, testNodes) << "For Cell " << test.first << std::endl;
    }
}

INSTANTIATE_TEST_SUITE_P(AxisymmetricTests, AxisymmetricMeshDescriptionFixture,
                         testing::Values(AxisymmetricMeshDescriptionParameters{
                             .description = std::make_shared<ablate::domain::descriptions::Axisymmetric>(std::vector<PetscReal>{0.0, 0.0, 0.0}, 1.0, 8, 4),
                             .expectedMeshDimension = 3,
                             .expectedNumberCells = 32,
                             .expectedNumberVertices = 45,
                             .expectedCellTypes = {{0, DM_POLYTOPE_TRI_PRISM}, {22, DM_POLYTOPE_TRI_PRISM}, {31, DM_POLYTOPE_TRI_PRISM}},
                             .expectedTopology = {{0, {0, 1, 2, 9, 11, 10}}, {7, {0, 8, 1, 9, 10, 17}}, {8, {9, 10, 11, 18, 20, 19}}, {15, {9, 17, 10, 18, 19, 26}}}}));
