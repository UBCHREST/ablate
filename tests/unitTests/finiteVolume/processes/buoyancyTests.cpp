#include <petsc.h>
#include <PetscTestFixture.hpp>
#include <vector>
#include "domain/boxMesh.hpp"
#include "domain/modifiers/ghostBoundaryCells.hpp"
#include "eos/mockEOS.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/processes/buoyancy.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/simpleFormula.hpp"

struct BuoyancyTestParameters {
    std::vector<double> buoyancy;
    std::shared_ptr<ablate::mathFunctions::FieldFunction> initialEuler;
    std::shared_ptr<ablate::mathFunctions::MathFunction> expectedSourceFunction;
};

class BuoyancyTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<BuoyancyTestParameters> {};

TEST_P(BuoyancyTestFixture, ShouldComputeCorrectFlux) {
    // arrange
    const auto& params = GetParam();

    // create a mock eos
    auto mockEOS = std::make_shared<ablateTesting::eos::MockEOS>();
    EXPECT_CALL(*mockEOS, GetSpeciesVariables).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(ablate::utilities::VectorUtilities::Empty<std::string>));
    EXPECT_CALL(*mockEOS, GetProgressVariables).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(ablate::utilities::VectorUtilities::Empty<std::string>));

    // Create a box mesh
    auto domain = std::make_shared<ablate::domain::BoxMesh>("testMesh",
                                                            std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>>{std::make_shared<ablate::finiteVolume::CompressibleFlowFields>(mockEOS)},
                                                            std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<ablate::domain::modifiers::GhostBoundaryCells>()},
                                                            std::vector<int>{5, 7, 10},
                                                            std::vector<double>{.0, .0, .0},
                                                            std::vector<double>{1, 1, 1});

    // Make a finite volume with only a buoyancy
    auto fvObject = std::make_shared<ablate::finiteVolume::FiniteVolumeSolver>(
        "testFV",
        ablate::domain::Region::ENTIREDOMAIN,
        nullptr /*options*/,
        std::vector<std::shared_ptr<ablate::finiteVolume::processes::Process>>{std::make_shared<ablate::finiteVolume::processes::Buoyancy>(GetParam().buoyancy)},
        std::vector<std::shared_ptr<ablate::finiteVolume::boundaryConditions::BoundaryCondition>>{});

    // init
    domain->InitializeSubDomains({fvObject}, std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>>{GetParam().initialEuler});
    fvObject->PreStep(nullptr);

    // get the solution vector local
    Vec locX, locF;
    DMGetLocalVector(domain->GetDM(), &locX) >> errorChecker;
    DMGlobalToLocal(domain->GetDM(), domain->GetSolutionVector(), INSERT_VALUES, locX) >> errorChecker;
    DMGetLocalVector(domain->GetDM(), &locF) >> errorChecker;
    VecSet(locF, 0.0);

    // act
    fvObject->ComputeRHSFunction(0.0, locX, locF) >> errorChecker;

    // assert
    // March over each cell
    PetscInt cStart, cEnd;
    const PetscInt* cells = nullptr;
    DMPlexGetSimplexOrBoxCells(fvObject->GetSubDomain().GetDM(), 0, &cStart, &cEnd);
    // get access to the array
    const PetscReal* locFArray;
    VecGetArrayRead(locF, &locFArray) >> errorChecker;

    for (PetscInt c = cStart; c < cEnd; ++c) {
        // if there is a cell array, use it, otherwise it is just c
        PetscBool boundary;
        DMIsBoundaryPoint(fvObject->GetSubDomain().GetDM(), c, &boundary);
        if (boundary) {
            continue;
        }

        // Get the raw data at this point, this check assumes the order the fields
        const PetscScalar* data;
        DMPlexPointLocalRead(fvObject->GetSubDomain().GetDM(), c, locFArray, &data) >> errorChecker;

        // compute the cell centroid
        PetscReal centroid[3];
        DMPlexComputeCellGeometryFVM(fvObject->GetSubDomain().GetDM(), c, nullptr, centroid, nullptr) >> errorChecker;

        // evaluate the expectedSource
        std::vector<double> expectedSource(5);
        GetParam().expectedSourceFunction->Eval(centroid, 3, 0.0, expectedSource);

        for (PetscInt d = 0; d < expectedSource.size(); d++) {
            ASSERT_NEAR(data[d], expectedSource[d], 1E-8) << "Expected buoyancy source is incorrect for component " << d << " in cell " << c;
        }
    }
    VecRestoreArrayRead(locF, &locFArray) >> errorChecker;

    // cleanup
    DMRestoreLocalVector(domain->GetDM(), &locX) >> errorChecker;
    DMRestoreLocalVector(domain->GetDM(), &locF) >> errorChecker;
}

INSTANTIATE_TEST_SUITE_P(BuoyancyTests, BuoyancyTestFixture,
                         testing::Values((BuoyancyTestParameters){
                             .buoyancy = {0.0, 0.0, -9.8},
                             .initialEuler = std::make_shared<ablate::mathFunctions::FieldFunction>(
                                 "euler", std::make_shared<ablate::mathFunctions::SimpleFormula>("1.1 + z, 0.0, (1.1 + z)*10*x, (1.1 + z)*20*y, (1.1 + z)*30*z")),
                             .expectedSourceFunction = std::make_shared<ablate::mathFunctions::SimpleFormula>("0.0,(30*z)*max(((1.1 + z)-1.6)*-9.8, 0.0), 0.0, 0.0, max(((1.1 + z)-1.6)*-9.8, 0.0)")}));
