#include <fstream>
#include "domain/descriptions/generatedAxis.hpp"
#include "gtest/gtest.h"

struct GeneratedAxisParameters {
    // Hold an mesh generator
    std::shared_ptr<ablate::domain::descriptions::GeneratedAxis> description;

    // Expected values
    std::vector<std::vector<PetscReal>> expectedVertices;
};

class GeneratedAxisFixture : public ::testing::TestWithParam<GeneratedAxisParameters> {};

TEST_P(GeneratedAxisFixture, ShouldBuildCorrectVertices) {
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

INSTANTIATE_TEST_SUITE_P(GeneratedAxisTests, GeneratedAxisFixture,
                         testing::Values(GeneratedAxisParameters{.description = std::make_shared<ablate::domain::descriptions::GeneratedAxis>(std::vector<PetscReal>{0.0, 0.0, 0.0}, 0.5, 2),
                                                                 .expectedVertices = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.5}}},

                                         GeneratedAxisParameters{.description = std::make_shared<ablate::domain::descriptions::GeneratedAxis>(std::vector<PetscReal>{0.1, 0.2, 0.3}, 0.5, 3),
                                                                 .expectedVertices = {{0.1, 0.2, 0.3}, {0.1, 0.2, 0.3 + 0.25}, {0.1, 0.2, 0.3 + 0.5}}},

                                         GeneratedAxisParameters{.description = std::make_shared<ablate::domain::descriptions::GeneratedAxis>(std::vector<PetscReal>{0.0, 0.0, 0.0}, 10, 11),
                                                                 .expectedVertices = {{0.0, 0.0, 0.0},
                                                                                      {0.0, 0.0, 1.0},
                                                                                      {0.0, 0.0, 2.0},
                                                                                      {0.0, 0.0, 3.0},
                                                                                      {0.0, 0.0, 4.0},
                                                                                      {0.0, 0.0, 5.0},
                                                                                      {0.0, 0.0, 6.0},
                                                                                      {0.0, 0.0, 7.0},
                                                                                      {0.0, 0.0, 8.0},
                                                                                      {0.0, 0.0, 9.0},
                                                                                      {0.0, 0.0, 10.0}}}));
