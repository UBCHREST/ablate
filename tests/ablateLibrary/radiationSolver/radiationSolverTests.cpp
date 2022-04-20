//
// Created by owen on 4/14/22.
//

#include <petsc.h>
#include <mathFunctions/functionFactory.hpp>
#include <memory>
#include "MpiTestFixture.hpp"
#include "builder.hpp"
#include "domain/boxMesh.hpp"
#include "domain/modifiers/distributeWithGhostCells.hpp"
#include "domain/modifiers/ghostBoundaryCells.hpp"
#include "eos/perfectGas.hpp"
#include "finiteVolume/boundaryConditions/ghost.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/compressibleFlowSolver.hpp"
#include "finiteVolume/fluxCalculator/ausm.hpp"
#include "gtest/gtest.h"
#include "monitors/timeStepMonitor.hpp"
#include "parameters/mapParameters.hpp"
#include "radiationSolver/radiationSolver.hpp"

struct RadiationTestParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    std::vector<int> meshFaces;
    std::vector<double> meshStart;
    std::vector<double> meshEnd;
    std::shared_ptr<ablate::mathFunctions::MathFunction> temperatureField;
    std::shared_ptr<ablate::mathFunctions::MathFunction> expectedResult;
};

class RadiationTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<RadiationTestParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

TEST_P(RadiationTestFixture, ShouldComputeCorrectSourceTerm) {
    StartWithMPI

        // initialize petsc and mpi
        PetscInitialize(argc, argv, NULL, "HELP") >> testErrorChecker;

        auto eos = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}}));

        // determine required fields for finite volume compressible flow, this will include euler and temperature
        std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptors = {std::make_shared<ablate::finiteVolume::CompressibleFlowFields>(eos)};

        auto domain =
            std::make_shared<ablate::domain::BoxMesh>("simpleMesh",
                                                      fieldDescriptors,
                                                      std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<ablate::domain::modifiers::DistributeWithGhostCells>(),
                                                                                                                        std::make_shared<ablate::domain::modifiers::GhostBoundaryCells>()},
                                                      GetParam().meshFaces,
                                                      GetParam().meshStart,
                                                      GetParam().meshEnd);

        // Setup the flow data
        auto parameters = std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"cfl", ".4"}});

        // Set the initial conditions for euler (not used, so set all to zero)
        auto initialConditionEuler = std::make_shared<ablate::mathFunctions::FieldFunction>("euler", std::make_shared<ablate::mathFunctions::ConstantValue>(0.0));

        // create a time stepper
        auto timeStepper = ablate::solver::TimeStepper("timeStepper", domain, {{"ts_max_steps", "0"}}, {}, {initialConditionEuler});

        // Create an instance of the radiationSolver
        auto radiationSolver =
            std::make_shared<ablate::finiteVolume::CompressibleFlowSolver>("compressibleShockTube",
                                                                           ablate::domain::Region::ENTIREDOMAIN,
                                                                           nullptr /*options*/,
                                                                           eos,
                                                                           parameters,
                                                                           nullptr /*transportModel*/,
                                                                           std::make_shared<ablate::finiteVolume::fluxCalculator::Ausm>(),
                                                                           std::vector<std::shared_ptr<ablate::finiteVolume::boundaryConditions::BoundaryCondition>>{} /*boundary conditions*/,
                                                                           true /*physics time step*/);

        // register the flowSolver with the timeStepper
        timeStepper.Register(radiationSolver, {std::make_shared<ablate::monitors::TimeStepMonitor>()});
        timeStepper.Solve();

        // force the aux variables of temperature to a known value
        auto auxVec = radiationSolver->GetSubDomain().GetAuxVector();
        auto auxFieldFunctions = {
            std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD, GetParam().temperatureField),
        };
        radiationSolver->GetSubDomain().ProjectFieldFunctionsToLocalVector(auxFieldFunctions, auxVec);

        // Setup the rhs for the test
        Vec rhs;
        DMGetLocalVector(domain->GetDM(), &rhs) >> testErrorChecker;
        VecZeroEntries(rhs) >> testErrorChecker;

        // Apply the rhs function for the radiation solver
        radiationSolver->RayTrace() >> testErrorChecker;

        // determine the euler field
        const auto& eulerFieldInfo = domain->GetField("euler");

        // For each cell, compare the rhs against the expected
        {
            // get the cell geometry
            Vec cellGeomVec;
            DM dmCell;
            const PetscScalar* cellGeomArray;
            DMPlexGetGeometryFVM(domain->GetDM(), nullptr, &cellGeomVec, nullptr) >> testErrorChecker;
            VecGetDM(cellGeomVec, &dmCell) >> testErrorChecker;
            VecGetArrayRead(cellGeomVec, &cellGeomArray) >> testErrorChecker;

            // extract the rhsArray
            const PetscScalar* rhsArray;
            VecGetArrayRead(rhs, &rhsArray) >> testErrorChecker;

            IS cellIS;
            PetscInt cellStart, cellEnd;
            const PetscInt* cells;
            radiationSolver->GetCellRange(cellIS, cellStart, cellEnd, cells);
            // March over each cell
            for (PetscInt c = cellStart; c < cellEnd; ++c) {
                // Get the cell location
                const PetscInt cell = cells ? cells[c] : c;

                // Get the cell center
                PetscFVCellGeom* cellGeom;
                DMPlexPointLocalRead(dmCell, cell, cellGeomArray, &cellGeom) >> testErrorChecker;

                // extract the result from the rhs
                PetscScalar* rhsValues;
                DMPlexPointLocalFieldRead(domain->GetDM(), cell, eulerFieldInfo.id, rhsArray, &rhsValues) >> testErrorChecker;
                PetscScalar actualResult = rhsArray[ablate::finiteVolume::CompressibleFlowFields::RHOE];

                // compute the expected result
                //PetscScalar expectedResult = GetParam().expectedResult->Eval(cellGeom->centroid, domain->GetDimensions(), 0.0);
                PetscScalar expectedResult = ablate::radiationSolver::RadiationSolver::ReallySolveParallelPlates(cellGeom->centroid[2]); //Compute the analytical solution at this z height.

                ASSERT_NEAR(expectedResult, actualResult, 1E-3) << "The actual result should be near the expected at cell " << cell << " [" << cellGeom->centroid[0] << ", " << cellGeom->centroid[1]
                                                                << ", " << cellGeom->centroid[2] << "]";
            }

            VecRestoreArrayRead(rhs, &rhsArray) >> testErrorChecker;
            VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> testErrorChecker;
        }

        DMRestoreLocalVector(domain->GetDM(), &rhs);
        exit(PetscFinalize());
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(RadiationTests, RadiationTestFixture,
                         testing::Values((RadiationTestParameters){.mpiTestParameter = {.testName = "1D uniform temperature", .nproc = 1},
                                                                   .meshFaces = { 3 , 3, 20},
                                                                   .meshStart = { 0 , 0 , -0.0105},
                                                                   .meshEnd = { 1 , 1 , 0.0105},
                                                                   .temperatureField = ablate::mathFunctions::Create("z < 0 ? (-6.349E6*z*z + 2000.0) : (-1.179E7*z*z + 2000.0)"),
                                                                   .expectedResult = ablate::mathFunctions::Create("x + y + z")}),
                         [](const testing::TestParamInfo<RadiationTestParameters>& info) { return info.param.mpiTestParameter.getTestName(); });