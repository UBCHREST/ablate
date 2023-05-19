#include "PetscTestFixture.hpp"
#include "domain/hdf5Initializer.hpp"
#include "gtest/gtest.h"
#include "mockField.hpp"
#include "utilities/vectorUtilities.hpp"

struct Hdf5InitializerTestPoint {
    std::vector<double> point;
    std::vector<double> expectedValue;
};

struct Hdf5InitializerTestParams {
    //! Path to the hdf5file
    std::filesystem::path hdf5File;

    //! test points
    std::map<std::string, std::vector<Hdf5InitializerTestPoint>> testPoints;
};

class Hdf5InitializerTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<Hdf5InitializerTestParams> {};

TEST_P(Hdf5InitializerTestFixture, ShouldComputeCorrectAnswerFromXYZ) {
    // arrange
    auto hdf5Initializer = std::make_shared<ablate::domain::Hdf5Initializer>(GetParam().hdf5File);

    // make the mock fields
    std::vector<ablate::domain::Field> mockFields;
    for (const auto& testPoint : GetParam().testPoints) {
        mockFields.push_back(ablateTesting::domain::MockField::Create(testPoint.first));
    }

    // Get the field functions
    auto fieldFunctions = hdf5Initializer->GetFieldFunctions(mockFields);

    // act/assert
    for (const auto& testSet : GetParam().testPoints) {
        // Get the fieldFunction
        auto fieldFunction = std::find_if(fieldFunctions.begin(), fieldFunctions.end(), [&testSet](const auto& ff) { return ff->GetName() == testSet.first; });
        if (fieldFunction == fieldFunctions.end()) {
            throw std::invalid_argument("Could not locate field function for " + testSet.first);
        }

        // Test each point
        auto mathFunction = fieldFunction->get()->GetFieldFunction();

        for (const auto& testPoint : testSet.second) {
            // setup input
            double x = testPoint.point[0];
            double y = testPoint.point[1];
            double z = testPoint.point.size() > 2 ? testPoint.point[2] : 0.0;

            // act/assert
            ASSERT_DOUBLE_EQ(testPoint.expectedValue.front(), mathFunction->Eval(x, y, z, NAN))
                << "Should be equal for for point " << ablate::utilities::VectorUtilities::Concatenate(testPoint.point) << " for file " << GetParam().hdf5File;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Hdf5InitializerTest, Hdf5InitializerTestFixture,
                         testing::Values((Hdf5InitializerTestParams){.hdf5File = "inputs/domain/initializer.2D.hdf5",
                                                                     .testPoints = {
                                                                         {"solution_fieldA", {{.point = {0.248599, 0.440033}, .expectedValue = {10.25,100.45}}}},
                                                                         {"solution_fieldB", {{.point = {0.253546, 0.445099}, .expectedValue = {0.1125}}}}

                                                                     }}));
