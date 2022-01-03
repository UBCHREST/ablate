#include <petsc.h>
#include <cmath>
#include <memory>
#include <vector>
#include "MpiTestFixture.hpp"
#include "PetscTestErrorChecker.hpp"
#include "convergenceTester.hpp"
#include "domain/boxMesh.hpp"
#include "domain/modifiers/distributeWithGhostCells.hpp"
#include "domain/modifiers/ghostBoundaryCells.hpp"
#include "eos/perfectGas.hpp"
#include "finiteVolume/boundaryConditions/essentialGhost.hpp"
#include "finiteVolume/boundaryConditions/ghost.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/compressibleFlowSolver.hpp"
#include "finiteVolume/fluxCalculator/ausm.hpp"
#include "finiteVolume/processes/flowProcess.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "monitors/solutionErrorMonitor.hpp"
#include "parameters/mapParameters.hpp"
#include "solver/directSolverTsInterface.hpp"

using namespace ablate;
using namespace ablate::finiteVolume;

struct CompressibleFlowEvAdvectionTestParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    PetscInt initialNx;
    int levels;
    std::shared_ptr<mathFunctions::MathFunction> eulerExact;
    std::shared_ptr<mathFunctions::MathFunction> densityEvExact;
    std::vector<PetscReal> expectedL2Convergence;
    std::vector<PetscReal> expectedLInfConvergence;
};

class CompressibleFlowEvAdvectionFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<CompressibleFlowEvAdvectionTestParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

TEST_P(CompressibleFlowEvAdvectionFixture, ShouldConvergeToExactSolution) {
    StartWithMPI
        // initialize petsc and mpi
        PetscInitialize(argc, argv, NULL, "HELP") >> testErrorChecker;

        PetscInt initialNx = GetParam().initialNx;

        testingResources::ConvergenceTester l2History("l2");
        testingResources::ConvergenceTester lInfHistory("lInf");

        // March over each level
        for (PetscInt l = 0; l < GetParam().levels; l++) {
            TS ts; /* timestepper */

            // Create a ts
            TSCreate(PETSC_COMM_WORLD, &ts) >> testErrorChecker;
            TSSetProblemType(ts, TS_NONLINEAR) >> testErrorChecker;
            TSSetType(ts, TSEULER) >> testErrorChecker;
            TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP) >> testErrorChecker;
            TSSetFromOptions(ts) >> testErrorChecker;

            // Create a mesh
            PetscInt nx1D = initialNx * PetscPowRealInt(2, l);

            PetscPrintf(PETSC_COMM_WORLD, "Running Calculation at Level %d (%dx%d)\n", l, nx1D, nx1D);

            // determine required fields for finite volume compressible flow
            auto eos = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287"}}));

            std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptors = {
                std::make_shared<ablate::finiteVolume::CompressibleFlowFields>(eos, std::vector<std::string>{"ev1", "ev2"})};

            auto mesh = std::make_shared<ablate::domain::BoxMesh>("simpleMesh",
                                                                  fieldDescriptors,
                                                                  std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::DistributeWithGhostCells>(),
                                                                                                                                    std::make_shared<domain::modifiers::GhostBoundaryCells>()},
                                                                  std::vector<int>{(int)nx1D, (int)nx1D},
                                                                  std::vector<double>{0.0, 0.0},
                                                                  std::vector<double>{.01, .01},
                                                                  std::vector<std::string>{} /*boundary*/,
                                                                  false /*simplex*/);
            // setup a flow parameters
            auto parameters = std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"cfl", "0.25"}});

            // setup solutions from the exact params
            auto exactEulerSolution = std::make_shared<mathFunctions::FieldFunction>(CompressibleFlowFields::EULER_FIELD, GetParam().eulerExact);
            auto evExactSolution = std::make_shared<mathFunctions::FieldFunction>(CompressibleFlowFields::DENSITY_EV_FIELD, GetParam().densityEvExact);

            auto boundaryConditions = std::vector<std::shared_ptr<finiteVolume::boundaryConditions::BoundaryCondition>>{
                std::make_shared<finiteVolume::boundaryConditions::EssentialGhost>("walls", std::vector<int>{1, 2, 3, 4}, exactEulerSolution),
                std::make_shared<finiteVolume::boundaryConditions::EssentialGhost>("walls", std::vector<int>{1, 2, 3, 4}, evExactSolution)};

            auto flowObject = std::make_shared<ablate::finiteVolume::CompressibleFlowSolver>("testFlow",
                                                                                             ablate::domain::Region::ENTIREDOMAIN,
                                                                                             nullptr /*options*/,
                                                                                             eos,
                                                                                             parameters,
                                                                                             nullptr /*transportModel*/,
                                                                                             std::make_shared<ablate::finiteVolume::fluxCalculator::Ausm>(),
                                                                                             std::vector<std::shared_ptr<processes::Process>>(),
                                                                                             boundaryConditions /*boundary conditions*/,
                                                                                             std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{exactEulerSolution, evExactSolution});

            mesh->InitializeSubDomains({flowObject}, std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{exactEulerSolution, evExactSolution});
            DMSetApplicationContext(mesh->GetDM(), flowObject.get());
            solver::DirectSolverTsInterface directSolverTsInterface(ts, flowObject);

            // Name the flow field
            PetscObjectSetName(((PetscObject)mesh->GetSolutionVector()), "Numerical Solution") >> testErrorChecker;

            // Setup the TS
            TSSetFromOptions(ts) >> testErrorChecker;

            // advance to the end time
            TSSolve(ts, mesh->GetSolutionVector()) >> testErrorChecker;

            PetscReal endTime;
            TSGetTime(ts, &endTime) >> testErrorChecker;

            // Get the L2 and LInf norms
            std::vector<PetscReal> l2Norm = ablate::monitors::SolutionErrorMonitor(ablate::monitors::SolutionErrorMonitor::Scope::COMPONENT, ablate::monitors::SolutionErrorMonitor::Norm::L2_NORM)
                                                .ComputeError(ts, endTime, mesh->GetSolutionVector());
            std::vector<PetscReal> lInfNorm = ablate::monitors::SolutionErrorMonitor(ablate::monitors::SolutionErrorMonitor::Scope::COMPONENT, ablate::monitors::SolutionErrorMonitor::Norm::LINF)
                                                  .ComputeError(ts, endTime, mesh->GetSolutionVector());

            // Store the residual into history
            const PetscReal h = 0.01 / nx1D;
            l2History.Record(h, l2Norm);
            lInfHistory.Record(h, lInfNorm);
            TSDestroy(&ts) >> testErrorChecker;
        }

        std::string l2Message;
        if (!l2History.CompareConvergenceRate(GetParam().expectedL2Convergence, l2Message)) {
            FAIL() << l2Message;
        }

        std::string lInfMessage;
        if (!lInfHistory.CompareConvergenceRate(GetParam().expectedLInfConvergence, lInfMessage)) {
            FAIL() << lInfMessage;
        }

        PetscErrorCode ierr = PetscFinalize();
        exit(ierr);

    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(CompressibleFlow, CompressibleFlowEvAdvectionFixture,
                         testing::Values(
                             (CompressibleFlowEvAdvectionTestParameters){
                                 .mpiTestParameter = {.testName = "ev advection",
                                                      .nproc = 1,
                                                      .arguments = "-dm_plex_separate_marker -petsclimiter_type none -ts_adapt_type none -automaticTimeStepCalculator off "
                                                                   "-euler_petscfv_type upwind -densityEV_petscfv_type upwind -ts_max_steps 50 -ts_dt 5e-05  "},
                                 .initialNx = 5,
                                 .levels = 4,
                                 .eulerExact = ablate::mathFunctions::Create("2.0, 500000, 8.0, 0.0"),
                                 .densityEvExact = ablate::mathFunctions::Create("2*.2*(1 + sin(2*_pi*(x-4*t)/.01))/2, 2*.3*(1 + sin(2*_pi*(x-4*t)/.01))/2"),
                                 .expectedL2Convergence = {NAN, NAN, NAN, NAN, 1, 1},
                                 .expectedLInfConvergence = {NAN, NAN, NAN, NAN, 1, 1}},
                             (CompressibleFlowEvAdvectionTestParameters){
                                 .mpiTestParameter = {.testName = "mpi ev advection",
                                                      .nproc = 2,
                                                      .arguments = "-dm_plex_separate_marker -dm_distribute -petsclimiter_type none -ts_adapt_type none -automaticTimeStepCalculator off "
                                                                   "-euler_petscfv_type upwind -densityEV_petscfv_type upwind -ts_max_steps 50 -ts_dt 5e-05  "},
                                 .initialNx = 5,
                                 .levels = 4,
                                 .eulerExact = ablate::mathFunctions::Create("2.0, 500000, 8.0, 0.0"),
                                 .densityEvExact = ablate::mathFunctions::Create("2*.2*(1 + sin(2*_pi*(x-4*t)/.01))/2, 2*.3*(1 + sin(2*_pi*(x-4*t)/.01))/2"),
                                 .expectedL2Convergence = {NAN, NAN, NAN, NAN, 1, 1},
                                 .expectedLInfConvergence = {NAN, NAN, NAN, NAN, 1, 1}}),
                         [](const testing::TestParamInfo<CompressibleFlowEvAdvectionTestParameters> &info) { return info.param.mpiTestParameter.getTestName(); });
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
