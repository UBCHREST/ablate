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

class Hdf5InitializerTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<Hdf5InitializerTestParams> {
   protected:
    // The hdf5Initializer after being setup
    std::shared_ptr<ablate::domain::Hdf5Initializer> hdf5Initializer;

    // the input field functions
    std::map<std::string, std::shared_ptr<ablate::mathFunctions::FieldFunction>> fieldFunctions;

    // Create the hdf5Initializer
    void SetUp() override {
        testingResources::PetscTestFixture::SetUp();

        hdf5Initializer = std::make_shared<ablate::domain::Hdf5Initializer>(GetParam().hdf5File);

        // make the mock fields
        std::vector<ablate::domain::Field> mockFields;
        for (const auto& testPoint : GetParam().testPoints) {
            mockFields.push_back(ablateTesting::domain::MockField::Create(testPoint.first));
        }

        // Get the field functions
        auto fieldFunctionsVector = hdf5Initializer->GetFieldFunctions(mockFields);

        // Convert to a map
        for (const auto& fieldFunction : fieldFunctionsVector) {
            fieldFunctions[fieldFunction->GetName()] = fieldFunction;
        }
    }
};

TEST_P(Hdf5InitializerTestFixture, ShouldComputeCorrectScalarFromXYZ) {
    // arrange
    // act/assert
    for (const auto& testSet : GetParam().testPoints) {
        // Get the fieldFunction
        auto fieldFunction = fieldFunctions[testSet.first];

        // Test each point
        auto mathFunction = fieldFunction->GetFieldFunction();

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

TEST_P(Hdf5InitializerTestFixture, ShouldComputeCorrectScalarFromCoord) {
    // arrange
    // act/assert
    for (const auto& testSet : GetParam().testPoints) {
        // Get the fieldFunction
        auto fieldFunction = fieldFunctions[testSet.first];

        // Test each point
        auto mathFunction = fieldFunction->GetFieldFunction();

        for (const auto& testPoint : testSet.second) {
            // act/assert
            ASSERT_DOUBLE_EQ(testPoint.expectedValue.front(), mathFunction->Eval(testPoint.point.data(), (PetscInt)testPoint.point.size(), NAN))
                << "Should be equal for for point " << ablate::utilities::VectorUtilities::Concatenate(testPoint.point) << " for file " << GetParam().hdf5File;
        }
    }
}

TEST_P(Hdf5InitializerTestFixture, ShouldComputeCorrectVectorFromXYZ) {
    // arrange
    // act/assert
    for (const auto& testSet : GetParam().testPoints) {
        // Get the fieldFunction
        auto fieldFunction = fieldFunctions[testSet.first];

        // Test each point
        auto mathFunction = fieldFunction->GetFieldFunction();

        for (const auto& testPoint : testSet.second) {
            // setup input
            double x = testPoint.point[0];
            double y = testPoint.point[1];
            double z = testPoint.point.size() > 2 ? testPoint.point[2] : 0.0;

            std::vector<double> result;

            // act
            mathFunction->Eval(x, y, z, NAN, result);

            // assert
            for (std::size_t i = 0; i < testPoint.expectedValue.size(); i++) {
                ASSERT_DOUBLE_EQ(testPoint.expectedValue[i], result[i])
                    << "Should be equal for for point " << ablate::utilities::VectorUtilities::Concatenate(testPoint.point) << " for file " << GetParam().hdf5File;
            }
        }
    }
}

TEST_P(Hdf5InitializerTestFixture, ShouldComputeCorrectVectorFromCoord) {
    // arrange
    // act/assert
    for (const auto& testSet : GetParam().testPoints) {
        // Get the fieldFunction
        auto fieldFunction = fieldFunctions[testSet.first];

        // Test each point
        auto mathFunction = fieldFunction->GetFieldFunction();

        for (const auto& testPoint : testSet.second) {
            std::vector<double> result;

            // act
            mathFunction->Eval(testPoint.point.data(), (PetscInt)testPoint.point.size(), NAN, result);

            // assert
            for (std::size_t i = 0; i < testPoint.expectedValue.size(); i++) {
                ASSERT_DOUBLE_EQ(testPoint.expectedValue[i], result[i])
                    << "Should be equal for for point " << ablate::utilities::VectorUtilities::Concatenate(testPoint.point) << " for file " << GetParam().hdf5File;
            }
        }
    }
}

TEST_P(Hdf5InitializerTestFixture, ShouldComputeCorrectAnswerPetscFunction) {
    // arrange
    // act/assert
    for (const auto& testSet : GetParam().testPoints) {
        // Get the fieldFunction
        auto fieldFunction = fieldFunctions[testSet.first];

        // Test each point
        auto mathFunction = fieldFunction->GetFieldFunction();
        auto functionPointer = mathFunction->GetPetscFunction();
        auto context = mathFunction->GetContext();

        for (const auto& testPoint : testSet.second) {
            std::vector<double> result(testPoint.expectedValue.size());

            // act
            functionPointer((PetscInt)testPoint.point.size(), NAN, testPoint.point.data(), (PetscInt)result.size(), result.data(), context);

            // assert
            for (std::size_t i = 0; i < testPoint.expectedValue.size(); i++) {
                ASSERT_DOUBLE_EQ(testPoint.expectedValue[i], result[i])
                    << "Should be equal for for point " << ablate::utilities::VectorUtilities::Concatenate(testPoint.point) << " for file " << GetParam().hdf5File;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Hdf5InitializerTest, Hdf5InitializerTestFixture,
                         testing::Values((Hdf5InitializerTestParams){.hdf5File = "inputs/domain/initializer.2D.hdf5",
                                                                     .testPoints = {{"solution_fieldA",
                                                                                     {{.point = {0.248599, 0.440033}, .expectedValue = {10.25, 100.45}},
                                                                                      {.point = {0.852413, 0.849201}, .expectedValue = {10.85, 100.85}},
                                                                                      {.point = {0.852413, 0.849201, .1}, .expectedValue = {10.85, 100.85}}}},
                                                                                    {"solution_fieldB",
                                                                                     {{.point = {0.253546, 0.445099}, .expectedValue = {0.1125}},
                                                                                      {.point = {0.162361, 0.8567}, .expectedValue = {0.1275}},
                                                                                      {.point = {0.758865, 0.253863}, .expectedValue = {0.1875}},
                                                                                      {.point = {0.872847, 0.840236}, .expectedValue = {0.7225}},
                                                                                      {.point = {0.872847, 0.840236, -1}, .expectedValue = {0.7225}}}}}},

                                         (Hdf5InitializerTestParams){.hdf5File = "inputs/domain/initializer.3D.hdf5",
                                                                     .testPoints = {{"solution_fieldA",
                                                                                     {{.point = {0.2, 0.682404, 0.260763}, .expectedValue = {10.15, 100.65, 0.4}},
                                                                                      {.point = {0.825138, 0.4, -0.231539}, .expectedValue = {10.85, 100.35, -0.4}},
                                                                                      {.point = {0.2, 0.862515, -0.366197}, .expectedValue = {10.15, 100.85, -0.8}}}},
                                                                                    {"solution_fieldB",
                                                                                     {{.point = {0.2, 0.682404, 0.260763}, .expectedValue = {0.0195}},
                                                                                      {.point = {0.825138, 0.4, -0.231539}, .expectedValue = {-0.0595}},
                                                                                      {.point = {0.2, 0.862515, -0.366197}, .expectedValue = {-0.051}}}}}}),
                         [](const testing::TestParamInfo<Hdf5InitializerTestParams>& info) { return testingResources::PetscTestFixture::SanitizeTestName(info.param.hdf5File.string()); });
