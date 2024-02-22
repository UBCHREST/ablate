#include <fstream>
#include "domain/descriptions/specifiedAxis.hpp"
#include "gtest/gtest.h"

struct SpecifiedAxisParameters {
    // Hold an mesh generator
    std::shared_ptr<ablate::domain::descriptions::SpecifiedAxis> description;

    // Expected values
    std::vector<std::vector<PetscReal>> expectedVertices;
};

class SpecifiedAxisFixture : public ::testing::TestWithParam<SpecifiedAxisParameters> {};

TEST_P(SpecifiedAxisFixture, ShouldBuildCorrectVertices) {
    // arrange
    auto description = GetParam().description;

    const auto& expectedVertices = GetParam().expectedVertices;

    // act
    auto const& numberVertices = description->GetNumberVertices();

    // assert
    ASSERT_EQ(numberVertices, expectedVertices.size());

    for (std::size_t v = 0; v < expectedVertices.size(); ++v) {
        // Build the testCellNodes
        std::vector<PetscReal> testVertex(expectedVertices[v].size());
        description->SetCoordinate((PetscInt)v, testVertex.data());

        for (std::size_t i = 0; i < testVertex.size(); ++i) {
            ASSERT_NEAR(expectedVertices[v][i], testVertex[i], 1E-8) << "For Node " << v << " index " << i << std::endl;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(SpecifiedAxisTests, SpecifiedAxisFixture,
                         testing::Values(SpecifiedAxisParameters{.description = std::make_shared<ablate::domain::descriptions::SpecifiedAxis>(std::vector<PetscReal>{1.0, 2.0, 5.0}),
                                                                 .expectedVertices = {{0.0, 0.0, 1.0}, {0.0, 0.0, 2.0}, {0.0, 0.0, 5.0}}},
                                         SpecifiedAxisParameters{
                                             .description = std::make_shared<ablate::domain::descriptions::SpecifiedAxis>(std::vector<PetscReal>{1.0, 2.0, 5.0, 5.5, 8.0},
                                                                                                                          std::vector<PetscReal>{2.0, 4.0, 10.0}),
                                             .expectedVertices = {{2.0, 4.0, 10.0 + 1.0}, {2.0, 4.0, 10.0 + 2.0}, {2.0, 4.0, 10.0 + 5.0}, {2.0, 4.0, 10.0 + 5.5}, {2.0, 4.0, 10.0 + 8.0}}}));
