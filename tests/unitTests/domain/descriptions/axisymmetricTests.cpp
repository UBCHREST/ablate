#include <fstream>
#include "domain/descriptions/axisymmetric.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"

struct AxisymmetricMeshDescriptionParameters {
    // Hold an mesh generator
    std::shared_ptr<ablate::domain::descriptions::Axisymmetric> description;

    // Expected values
    PetscInt expectedMeshDimension;
    PetscInt expectedNumberCells;
    PetscInt expectedNumberVertices;
    std::map<PetscInt, DMPolytopeType> expectedCellTypes;
    std::map<PetscInt, std::vector<PetscInt>> expectedTopology;
    std::map<PetscInt, std::vector<PetscReal>> expectedVertices;
    std::map<std::shared_ptr<ablate::domain::Region>, std::vector<std::set<PetscInt>>> expectedBoundaries;
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

TEST_P(AxisymmetricMeshDescriptionFixture, ShouldBuildCorrectVertices) {
    // arrange
    auto description = GetParam().description;
    // act
    // assert
    for (const auto& test : GetParam().expectedVertices) {
        // Build the testCellNodes
        std::vector<PetscReal> testVertex(test.second.size());
        description->SetCoordinate(test.first, testVertex.data());

        for (std::size_t i = 0; i < testVertex.size(); ++i) {
            ASSERT_NEAR(test.second[i], testVertex[i], 1E-8) << "For Node " << test.first << " index " << i << std::endl;
        }
    }
}

TEST_P(AxisymmetricMeshDescriptionFixture, ShouldIdentifyCorrectBoundaries) {
    // arrange
    auto description = GetParam().description;
    // act
    // assert
    for (const auto& test : GetParam().expectedBoundaries) {
        // march over each node set
        for (const auto& nodeSet : test.second) {
            auto region = description->GetRegion(nodeSet);

            // build the error nodes if needed
            std::stringstream nodes;
            for (const auto& n : nodeSet) {
                nodes << n << " ";
            }
            if (test.first) {
                if (!region) {
                    FAIL() << "Should be region for nodes: " << nodes.str() << std::endl;
                }
                ASSERT_EQ(*test.first, *region) << "Should be equal for nodes: " << nodes.str() << std::endl;
            } else {
                ASSERT_EQ(region, nullptr) << "Should be null " << nodes.str() << std::endl;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(AxisymmetricTests, AxisymmetricMeshDescriptionFixture,
                         testing::Values(
                             // 1 slice, 1 shell
                             AxisymmetricMeshDescriptionParameters{
                                 .description = std::make_shared<ablate::domain::descriptions::Axisymmetric>(std::vector<PetscReal>{0.0, 0.0, 0.0}, 0.5, ablate::mathFunctions::Create(1.0), 8, 1, 1),
                                 .expectedMeshDimension = 3,
                                 .expectedNumberCells = 8,
                                 .expectedNumberVertices = 18,
                                 .expectedCellTypes = {{0, DM_POLYTOPE_TRI_PRISM}, {4, DM_POLYTOPE_TRI_PRISM}, {7, DM_POLYTOPE_TRI_PRISM}},
                                 .expectedTopology = {{0, {0, 2, 3, 1, 11, 10}}, {7, {0, 9, 2, 1, 10, 17}}},
                                 .expectedVertices = {{0, {0.0, 0.0, 0.0}}, {2, {1.0, 0.0, 0.0}}, {4, {0.0, 1.0, 0.0}}, {1, {0.0, 0.0, 0.5}}, {10, {1.0, 0.0, 0.5}}, {12, {0.0, 1.0, 0.5}}},
                                 .expectedBoundaries = {{ablate::domain::Region::ENTIREDOMAIN, {{0, 3, 1, 11}}},
                                                        {std::make_shared<ablate::domain::Region>("outerShell"), {{2, 9, 10, 17}, {5, 4, 12, 13}}},
                                                        {std::make_shared<ablate::domain::Region>("lowerCap"), {{0, 2, 3}, {0, 5, 6}}},
                                                        {std::make_shared<ablate::domain::Region>("upperCap"), {{1, 16, 17}, {1, 17, 10}}}}},
                             // 2 slices, 1 shell
                             AxisymmetricMeshDescriptionParameters{
                                 .description = std::make_shared<ablate::domain::descriptions::Axisymmetric>(std::vector<PetscReal>{0.0, 0.0, 0.0}, 0.5, ablate::mathFunctions::Create(1.0), 8, 2, 1),
                                 .expectedMeshDimension = 3,
                                 .expectedNumberCells = 16,
                                 .expectedNumberVertices = 27,
                                 .expectedCellTypes = {{0, DM_POLYTOPE_TRI_PRISM}, {7, DM_POLYTOPE_TRI_PRISM}, {8, DM_POLYTOPE_TRI_PRISM}, {15, DM_POLYTOPE_TRI_PRISM}},
                                 .expectedTopology = {{0, {0, 3, 4, 1, 12, 11}}, {7, {0, 10, 3, 1, 11, 18}}, {8, {1, 11, 12, 2, 20, 19}}, {15, {1, 18, 11, 2, 19, 26}}},
                                 .expectedVertices = {{0, {0.0, 0.0, 0.0}},
                                                      {3, {1.0, 0.0, 0.0}},
                                                      {5, {0.0, 1.0, 0.0}},
                                                      {1, {0.0, 0.0, 0.25}},
                                                      {11, {1.0, 0.0, 0.25}},
                                                      {13, {0.0, 1.0, 0.25}},
                                                      {2, {0.0, 0.0, 0.5}},
                                                      {19, {1.0, 0.0, 0.5}},
                                                      {21, {0.0, 1.0, 0.5}}},
                                 .expectedBoundaries = {{ablate::domain::Region::ENTIREDOMAIN, {{1, 12, 13}}},
                                                        {std::make_shared<ablate::domain::Region>("outerShell"), {{3, 4, 11, 12}, {19, 25, 11, 18}}},
                                                        {std::make_shared<ablate::domain::Region>("lowerCap"), {{0, 3, 4}, {0, 10, 3}}},
                                                        {std::make_shared<ablate::domain::Region>("upperCap"), {{2, 19, 20}, {2, 19, 26}}}}},
                             // 1 slice, 2 shells
                             AxisymmetricMeshDescriptionParameters{
                                 .description = std::make_shared<ablate::domain::descriptions::Axisymmetric>(std::vector<PetscReal>{0.0, 0.0, 0.0}, 0.5, ablate::mathFunctions::Create(1.0), 8, 1, 2),
                                 .expectedMeshDimension = 3,
                                 .expectedNumberCells = 16,
                                 .expectedNumberVertices = 34,
                                 .expectedCellTypes = {{0, DM_POLYTOPE_TRI_PRISM}, {7, DM_POLYTOPE_TRI_PRISM}, {8, DM_POLYTOPE_HEXAHEDRON}, {15, DM_POLYTOPE_HEXAHEDRON}},
                                 .expectedTopology = {{0, {0, 2, 3, 1, 11, 10}}, {7, {0, 9, 2, 1, 10, 17}}, {8, {2, 18, 19, 3, 10, 11, 27, 26}}, {15, {9, 25, 18, 2, 17, 10, 26, 33}}},
                                 .expectedVertices = {{0, {0.0, 0.0, 0.0}},
                                                      {2, {0.5, 0.0, 0.0}},
                                                      {4, {0.0, 0.5, 0.0}},
                                                      {18, {1.0, 0.0, 0.0}},
                                                      {20, {0.0, 1.0, 0.0}},
                                                      {1, {0.0, 0.0, 0.5}},
                                                      {10, {0.5, 0.0, 0.5}},
                                                      {12, {0.0, 0.5, 0.5}},
                                                      {26, {1.0, 0.0, 0.5}},
                                                      {28, {0.0, 1.0, 0.5}}},
                                 .expectedBoundaries = {{ablate::domain::Region::ENTIREDOMAIN, {{0, 3, 11}}},
                                                        {std::make_shared<ablate::domain::Region>("outerShell"), {{18, 19, 26, 27}, {26, 33, 18, 25}, {23, 24, 31, 32}}},
                                                        {std::make_shared<ablate::domain::Region>("lowerCap"), {{0, 2, 3}, {0, 9, 2}, {2, 19, 19, 3}, {2, 18, 9, 25}}},
                                                        {std::make_shared<ablate::domain::Region>("upperCap"), {{1, 10, 11}, {1, 17, 10}, {10, 26, 11, 27}, {10, 26, 17, 33}}}}},
                             // 2 slice, 2 shells
                             AxisymmetricMeshDescriptionParameters{
                                 .description = std::make_shared<ablate::domain::descriptions::Axisymmetric>(std::vector<PetscReal>{0.0, 0.0, 0.0}, 1.0, ablate::mathFunctions::Create(1.0), 8, 2, 2),
                                 .expectedMeshDimension = 3,
                                 .expectedNumberCells = 32,
                                 .expectedNumberVertices = 51,
                                 .expectedCellTypes = {{0, DM_POLYTOPE_TRI_PRISM},
                                                       {7, DM_POLYTOPE_TRI_PRISM},
                                                       {8, DM_POLYTOPE_TRI_PRISM},
                                                       {15, DM_POLYTOPE_TRI_PRISM},
                                                       {16, DM_POLYTOPE_HEXAHEDRON},
                                                       {23, DM_POLYTOPE_HEXAHEDRON},
                                                       {23, DM_POLYTOPE_HEXAHEDRON},
                                                       {31, DM_POLYTOPE_HEXAHEDRON}},
                                 .expectedTopology = {{0, {0, 3, 4, 1, 12, 11}},
                                                      {7, {0, 10, 3, 1, 11, 18}},
                                                      {16, {3, 27, 28, 4, 11, 12, 36, 35}},
                                                      {23, {10, 34, 27, 3, 18, 11, 35, 42}},
                                                      {8, {1, 11, 12, 2, 20, 19}},
                                                      {15, {1, 18, 11, 2, 19, 26}},
                                                      {24, {11, 35, 36, 12, 19, 20, 44, 43}},
                                                      {31, {18, 42, 35, 11, 26, 19, 43, 50}}},
                                 .expectedVertices = {{0, {0.0, 0.0, 0.0}},
                                                      {3, {0.5, 0.0, 0.0}},
                                                      {5, {0.0, 0.5, 0.0}},
                                                      {27, {1.0, 0.0, 0.0}},
                                                      {29, {0.0, 1.0, 0.0}},
                                                      {1, {0.0, 0.0, 0.5}},
                                                      {11, {0.5, 0.0, 0.5}},
                                                      {13, {0.0, 0.5, 0.5}},
                                                      {35, {1.0, 0.0, 0.5}},
                                                      {37, {0.0, 1.0, 0.5}},
                                                      {2, {0.0, 0.0, 1.0}},
                                                      {19, {0.5, 0.0, 1.0}},
                                                      {21, {0.0, 0.5, 1.0}},
                                                      {43, {1.0, 0.0, 1.0}},
                                                      {45, {0.0, 1.0, 1.0}}},
                                 .expectedBoundaries = {{ablate::domain::Region::ENTIREDOMAIN, {{1, 11, 12}}},
                                                        {std::make_shared<ablate::domain::Region>("outerShell"), {{28, 27, 36, 35}, {44, 43, 36, 35}, {48, 49, 30, 41}}},
                                                        {std::make_shared<ablate::domain::Region>("lowerCap"), {{0, 3, 4}, {0, 10, 3}, {3, 4, 28, 27}, {10, 3, 27, 34}}},
                                                        {std::make_shared<ablate::domain::Region>("upperCap"), {{2, 19, 20}, {2, 19, 26}, {26, 50, 43, 19}}}}},
                             // 2 slice, 2 shells, variable radius
                             AxisymmetricMeshDescriptionParameters{
                                 .description = std::make_shared<ablate::domain::descriptions::Axisymmetric>(std::vector<PetscReal>{0.0, 0.0, 0.0}, 2.0, ablate::mathFunctions::Create(".1*z+ 0.5"), 8,
                                                                                                             2, 2),
                                 .expectedMeshDimension = 3,
                                 .expectedNumberCells = 32,
                                 .expectedNumberVertices = 51,
                                 .expectedCellTypes = {{0, DM_POLYTOPE_TRI_PRISM},
                                                       {7, DM_POLYTOPE_TRI_PRISM},
                                                       {8, DM_POLYTOPE_TRI_PRISM},
                                                       {15, DM_POLYTOPE_TRI_PRISM},
                                                       {16, DM_POLYTOPE_HEXAHEDRON},
                                                       {23, DM_POLYTOPE_HEXAHEDRON},
                                                       {23, DM_POLYTOPE_HEXAHEDRON},
                                                       {31, DM_POLYTOPE_HEXAHEDRON}},
                                 .expectedTopology = {{0, {0, 3, 4, 1, 12, 11}},
                                                      {7, {0, 10, 3, 1, 11, 18}},
                                                      {16, {3, 27, 28, 4, 11, 12, 36, 35}},
                                                      {23, {10, 34, 27, 3, 18, 11, 35, 42}},
                                                      {8, {1, 11, 12, 2, 20, 19}},
                                                      {15, {1, 18, 11, 2, 19, 26}},
                                                      {24, {11, 35, 36, 12, 19, 20, 44, 43}},
                                                      {31, {18, 42, 35, 11, 26, 19, 43, 50}}},
                                 .expectedVertices = {{0, {0.0, 0.0, 0.0}},
                                                      {3, {0.25, 0.0, 0.0}},
                                                      {5, {0.0, 0.25, 0.0}},
                                                      {27, {0.5, 0.0, 0.0}},
                                                      {29, {0.0, 0.5, 0.0}},
                                                      {1, {0.0, 0.0, 1.0}},
                                                      {11, {0.3, 0.0, 1.0}},
                                                      {13, {0.0, 0.3, 1.0}},
                                                      {35, {0.6, 0.0, 1.0}},
                                                      {37, {0.0, 0.6, 1.0}},
                                                      {2, {0.0, 0.0, 2.0}},
                                                      {19, {0.35, 0.0, 2.0}},
                                                      {21, {0.0, 0.35, 2.0}},
                                                      {43, {0.7, 0.0, 2.0}},
                                                      {45, {0.0, 0.7, 2.0}}},
                                 .expectedBoundaries = {{ablate::domain::Region::ENTIREDOMAIN, {{1, 11, 12}}},
                                                        {std::make_shared<ablate::domain::Region>("outerShell"), {{28, 27, 36, 35}, {44, 43, 36, 35}, {48, 49, 30, 41}}},
                                                        {std::make_shared<ablate::domain::Region>("lowerCap"), {{0, 3, 4}, {0, 10, 3}, {3, 4, 28, 27}, {10, 3, 27, 34}}},
                                                        {std::make_shared<ablate::domain::Region>("upperCap"), {{2, 19, 20}, {2, 19, 26}, {26, 50, 43, 19}}}}}));
