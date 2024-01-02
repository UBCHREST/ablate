#include <fstream>
#include "domain/descriptions/axisymmetric.hpp"
#include "gtest/gtest.h"

struct AxisymmetricMeshGeneratorParameters {
    // Hold an mesh generator
    std::shared_ptr<ablate::domain::descriptions::Axisymmetric> meshGenerator;

    // Expected values
    PetscInt expectedNumberCells;
    PetscInt expectedNumberVertices;
};

class AxisymmetricMeshGeneratorFixture : public ::testing::TestWithParam<AxisymmetricMeshGeneratorParameters> {};

TEST_P(AxisymmetricMeshGeneratorFixture, ShouldComputeCorrectNumberOfCellsAndVertices) {
    // arrange
    auto meshGenerator = GetParam().meshGenerator;
    // act
    // assert
    ASSERT_EQ(GetParam().expectedNumberCells, meshGenerator->GetNumberCells());
    ASSERT_EQ(GetParam().expectedNumberVertices, meshGenerator->GetNumberVertices());
}

INSTANTIATE_TEST_SUITE_P(AxisymmetricTests, AxisymmetricMeshGeneratorFixture,
                         testing::Values(AxisymmetricMeshGeneratorParameters{
                             .meshGenerator = std::make_shared<ablate::domain::descriptions::Axisymmetric>(std::vector<PetscReal>{0.0, 0.0, 0.0}, std::vector<PetscReal>{0.0, 0.0, 1.0}, 8, 4),
                             .expectedNumberCells = 32,
                             .expectedNumberVertices = 45}));
