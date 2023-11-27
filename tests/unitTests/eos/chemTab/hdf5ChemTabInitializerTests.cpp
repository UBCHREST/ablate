#include "domain/hdf5Initializer.hpp"
#include "domain/mockField.hpp"
#include "eos/chemTab.hpp"
#include "eos/chemTab/hdf5ChemTabInitializer.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "gtest/gtest.h"
#include "petscTestFixture.hpp"
#include "tensorFlowCheck.hpp"
#include "utilities/vectorUtilities.hpp"

struct Hdf5ChemTabInitializerTestParams {
    //! Path to the hdf5file
    std::filesystem::path hdf5File;

    //! Use a test chemTab model
    std::filesystem::path chemTabModel;

    //! test points
    std::vector<std::vector<PetscReal>> testPoints;
};

class Hdf5ChemTabInitializerTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<Hdf5ChemTabInitializerTestParams> {
   protected:
    // The hdf5Initializer after being setup used for reference
    std::shared_ptr<ablate::domain::Hdf5Initializer> referenceHdf5Initializer;

    // The hdf5Initializer after being setup used for reference
    std::shared_ptr<ablate::eos::chemTab::Hdf5ChemTabInitializer> hdf5ChemTabInitializer;

    // store the reference and chemTab field initializers
    std::shared_ptr<ablate::mathFunctions::FieldFunction> referenceEulerFunction;
    std::shared_ptr<ablate::mathFunctions::FieldFunction> referenceDensityYiFunction;
    std::shared_ptr<ablate::mathFunctions::FieldFunction> chemTabEulerFunction;
    std::shared_ptr<ablate::mathFunctions::FieldFunction> chemTabDensityProgressVariableFunction;

    std::shared_ptr<ablate::eos::ChemTab> chemTab;

    // Create the hdf5Initializer
    void SetUp() override {
        testingResources::PetscTestFixture::SetUp();

        // create the chemTab model
        chemTab = std::make_shared<ablate::eos::ChemTab>(GetParam().chemTabModel);

        referenceHdf5Initializer = std::make_shared<ablate::domain::Hdf5Initializer>(GetParam().hdf5File);
        hdf5ChemTabInitializer = std::make_shared<ablate::eos::chemTab::Hdf5ChemTabInitializer>(GetParam().hdf5File, chemTab);

        // Get the field functions
        referenceEulerFunction = referenceHdf5Initializer->GetFieldFunctions({ablateTesting::domain::MockField::Create(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD)}).front();
        referenceDensityYiFunction = referenceHdf5Initializer->GetFieldFunctions({ablateTesting::domain::MockField::Create(ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD)}).front();
        chemTabEulerFunction = hdf5ChemTabInitializer->GetFieldFunctions({ablateTesting::domain::MockField::Create(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD)}).front();
        chemTabDensityProgressVariableFunction =
            hdf5ChemTabInitializer->GetFieldFunctions({ablateTesting::domain::MockField::Create(ablate::finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD)}).front();
    }
};

TEST_P(Hdf5ChemTabInitializerTestFixture, ShouldComputeCorrectVectorFromCoord) {
    ONLY_WITH_TENSORFLOW_CHECK;

    // march over each test point
    for (const auto& testPoint : GetParam().testPoints) {
        // arrange
        // size up the result fields
        std::vector<PetscReal> referenceEuler(ablate::finiteVolume::CompressibleFlowFields::EulerComponents::RHOU + testPoint.size());
        std::vector<PetscReal> referenceYis(chemTab->GetSpeciesNames().size());
        std::vector<PetscReal> referenceDensityProgress(chemTab->GetProgressVariables().size());

        std::vector<PetscReal> chemTabEuler(ablate::finiteVolume::CompressibleFlowFields::EulerComponents::RHOU + testPoint.size());
        std::vector<PetscReal> chemTabDensityProgress(chemTab->GetProgressVariables().size());

        // compute the expected referenceDensityProgress
        referenceEulerFunction->GetFieldFunction()->Eval(testPoint.data(), (int)testPoint.size(), NAN, referenceEuler);
        referenceDensityYiFunction->GetFieldFunction()->Eval(testPoint.data(), (int)testPoint.size(), NAN, referenceYis);

        // scale the reference yi by density
        for (auto& yi : referenceYis) {
            yi /= referenceEuler[ablate::finiteVolume::CompressibleFlowFields::EulerComponents::RHO];
        }

        // compute the expcted progress variable
        chemTab->ComputeProgressVariables(referenceYis, referenceDensityProgress);
        for (auto& densityProgress : referenceDensityProgress) {
            densityProgress *= referenceEuler[ablate::finiteVolume::CompressibleFlowFields::EulerComponents::RHO];
        }

        // act
        // compute the expected chemTabDensityProgress and chemTabEuler
        chemTabEulerFunction->GetFieldFunction()->Eval(testPoint.data(), (int)testPoint.size(), NAN, chemTabEuler);
        chemTabDensityProgressVariableFunction->GetFieldFunction()->Eval(testPoint.data(), (int)testPoint.size(), NAN, chemTabDensityProgress);

        // assert
        for (std::size_t i = 0; i < referenceEuler.size(); i++) {
            ASSERT_DOUBLE_EQ(referenceEuler[i], chemTabEuler[i]) << "Euler should be equal be equal for for point " << ablate::utilities::VectorUtilities::Concatenate(testPoint) << " for file "
                                                                 << GetParam().hdf5File;
        }

        for (std::size_t i = 0; i < referenceDensityProgress.size(); i++) {
            ASSERT_DOUBLE_EQ(referenceDensityProgress[i], chemTabDensityProgress[i])
                << "DensityProgress should be equal be equal for for point " << ablate::utilities::VectorUtilities::Concatenate(testPoint) << " for file " << GetParam().hdf5File;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Hdf5ChemTabInitializerTest, Hdf5ChemTabInitializerTestFixture,
                         testing::Values((Hdf5ChemTabInitializerTestParams){
                             .hdf5File = "inputs/eos/chemTab/initializer.1D.yi.hdf5", .chemTabModel = "inputs/eos/chemTabTestModel_1", .testPoints = {{0.0015}, {0.005}, {0.008}}}),
                         [](const testing::TestParamInfo<Hdf5ChemTabInitializerTestParams>& info) { return testingResources::PetscTestFixture::SanitizeTestName(info.param.hdf5File.string()); });
