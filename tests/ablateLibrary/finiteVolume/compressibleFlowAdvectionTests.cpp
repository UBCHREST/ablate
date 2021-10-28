#include <petsc.h>
#include <cmath>
#include <domain/modifiers/distributeWithGhostCells.hpp>
#include <domain/modifiers/ghostBoundaryCells.hpp>
#include <finiteVolume/boundaryConditions/essentialGhost.hpp>
#include <memory>
#include <monitors/solutionErrorMonitor.hpp>
#include <solver/directSolverTsInterface.hpp>
#include <vector>
#include "MpiTestFixture.hpp"
#include "PetscTestErrorChecker.hpp"
#include "domain/boxMesh.hpp"
#include "eos/perfectGas.hpp"
#include "finiteVolume/boundaryConditions/ghost.hpp"
#include "finiteVolume/compressibleFlow.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "parameters/mapParameters.hpp"

using namespace ablate;

struct CompressibleFlowAdvectionTestParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    PetscInt initialNx;
    int levels;
    std::shared_ptr<mathFunctions::MathFunction> eulerExact;
    std::shared_ptr<mathFunctions::MathFunction> densityYiExact;
    std::vector<PetscReal> expectedL2Convergence;
    std::vector<PetscReal> expectedLInfConvergence;
};

class CompressibleFlowAdvectionFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<CompressibleFlowAdvectionTestParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

TEST_P(CompressibleFlowAdvectionFixture, ShouldConvergeToExactSolution) {
    StartWithMPI
        // initialize petsc and mpi
        PetscInitialize(argc, argv, NULL, "HELP") >> testErrorChecker;

        PetscInt blockSize = 4 + 3;
        PetscInt initialNx = GetParam().initialNx;

        std::vector<PetscReal> hHistory;
        std::vector<std::vector<PetscReal>> l2History(blockSize);
        std::vector<std::vector<PetscReal>> lInfHistory(blockSize);

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

            auto mesh = std::make_shared<ablate::domain::BoxMesh>("simpleMesh",
                                                                  std::vector<int>{(int)nx1D, (int)nx1D},
                                                                  std::vector<double>{0.0, 0.0},
                                                                  std::vector<double>{.01, .01},
                                                                  std::vector<std::string>{} /*boundary*/,
                                                                  false /*simplex*/,
                                                                  nullptr /*options*/,
                                                                  std::vector<std::shared_ptr<ablate::domain::modifier::Modifier>>{std::make_shared<domain::modifier::DistributeWithGhostCells>(),
                                                                                                                                   std::make_shared<domain::modifier::GhostBoundaryCells>()});
            // setup a flow parameters
            auto parameters = std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"cfl", "0.25"}});

            auto eos = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287"}}),
                                                                 std::vector<std::string>{"O2", "H2O", "N2"});

            // setup solutions from the exact params
            auto exactEulerSolution = std::make_shared<mathFunctions::FieldFunction>("euler", GetParam().eulerExact);
            auto yiExactSolution = std::make_shared<mathFunctions::FieldFunction>("densityYi", GetParam().densityYiExact);

            auto boundaryConditions = std::vector<std::shared_ptr<finiteVolume::boundaryConditions::BoundaryCondition>>{
                std::make_shared<finiteVolume::boundaryConditions::EssentialGhost>("walls", std::vector<int>{1, 2, 3, 4}, exactEulerSolution),
                std::make_shared<finiteVolume::boundaryConditions::EssentialGhost>("walls", std::vector<int>{1, 2, 3, 4}, yiExactSolution)};

            auto flowObject =
                std::make_shared<ablate::finiteVolume::CompressibleFlow>("testFlow",
                                                                         ablate::domain::Region::ENTIREDOMAIN,
                                                                         nullptr /*options*/,
                                                                         eos,
                                                                         parameters,
                                                                         nullptr /*transportModel*/,
                                                                         nullptr,
                                                                         std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{exactEulerSolution, yiExactSolution} /*initialization*/,
                                                                         boundaryConditions /*boundary conditions*/,
                                                                         std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{exactEulerSolution, yiExactSolution});

            mesh->InitializeSubDomains({flowObject});
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

            // print the results to help with debug
            auto l2String = PrintVector(l2Norm, "%2.3g");
            PetscPrintf(PETSC_COMM_WORLD, "\tL_2 Error: %s\n", l2String.c_str()) >> testErrorChecker;

            auto lInfString = PrintVector(lInfNorm, "%2.3g");
            PetscPrintf(PETSC_COMM_WORLD, "\tL_2 L_Inf: %s\n", lInfString.c_str()) >> testErrorChecker;

            // Store the residual into history
            hHistory.push_back(PetscLog10Real(0.01 / nx1D));
            for (auto b = 0; b < blockSize; b++) {
                l2History[b].push_back(PetscLog10Real(l2Norm[b]));
                lInfHistory[b].push_back(PetscLog10Real(lInfNorm[b]));
            }
            TSDestroy(&ts) >> testErrorChecker;
        }

        // Fit each component and output
        for (auto b = 0; b < blockSize; b++) {
            PetscReal l2Slope;
            PetscReal l2Intercept;
            PetscLinearRegression(hHistory.size(), &hHistory[0], &l2History[b][0], &l2Slope, &l2Intercept) >> testErrorChecker;

            PetscReal lInfSlope;
            PetscReal lInfIntercept;
            PetscLinearRegression(hHistory.size(), &hHistory[0], &lInfHistory[b][0], &lInfSlope, &lInfIntercept) >> testErrorChecker;

            PetscPrintf(PETSC_COMM_WORLD, "Convergence[%d]: L2 %2.3g LInf %2.3g \n", b, l2Slope, lInfSlope) >> testErrorChecker;

            if (std::isnan(GetParam().expectedL2Convergence[b])) {
                ASSERT_TRUE(std::isnan(l2Slope)) << "incorrect L2 convergence order for component[" << b << "]";
            } else {
                ASSERT_NEAR(l2Slope, GetParam().expectedL2Convergence[b], 0.2) << "incorrect L2 convergence order for component[" << b << "]";
            }
            if (std::isnan(GetParam().expectedLInfConvergence[b])) {
                ASSERT_TRUE(std::isnan(lInfSlope)) << "incorrect LInf convergence order for component[" << b << "]";
            } else {
                ASSERT_NEAR(lInfSlope, GetParam().expectedLInfConvergence[b], 0.2) << "incorrect LInf convergence order for component[" << b << "]";
            }
        }
        PetscErrorCode ierr = PetscFinalize();
        exit(ierr);

    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(CompressibleFlow, CompressibleFlowAdvectionFixture,
                         testing::Values(
                             (CompressibleFlowAdvectionTestParameters){
                                 .mpiTestParameter = {.testName = "yi advection",
                                                      .nproc = 1,
                                                      .arguments = "-dm_plex_separate_marker -petsclimiter_type none -ts_adapt_type none -automaticTimeStepCalculator off "
                                                                   "-eulerpetscfv_type upwind -densityYipetscfv_type upwind -ts_max_steps 50 -ts_dt 5e-05  "},
                                 .initialNx = 5,
                                 .levels = 4,
                                 .eulerExact = ablate::mathFunctions::Create("2.0, 500000, 8.0, 0.0"),
                                 .densityYiExact = ablate::mathFunctions::Create("2*.2*(1 + sin(2*_pi*(x-4*t)/.01))/2, 2*.3*(1 + sin(2*_pi*(x-4*t)/.01))/2, 2*(1-.5*(1 + sin(2*_pi*(x-4*t)/.01))/2)"),
                                 .expectedL2Convergence = {NAN, NAN, NAN, NAN, 1, 1, 1},
                                 .expectedLInfConvergence = {NAN, NAN, NAN, NAN, 1, 1, 1}},
                             (CompressibleFlowAdvectionTestParameters){
                                 .mpiTestParameter = {.testName = "mpi yi advection",
                                                      .nproc = 2,
                                                      .arguments = "-dm_plex_separate_marker -dm_distribute -petsclimiter_type none -ts_adapt_type none -automaticTimeStepCalculator off "
                                                                   "-eulerpetscfv_type upwind -densityYipetscfv_type upwind -ts_max_steps 50 -ts_dt 5e-05  "},
                                 .initialNx = 5,
                                 .levels = 4,
                                 .eulerExact = ablate::mathFunctions::Create("2.0, 500000, 8.0, 0.0"),
                                 .densityYiExact = ablate::mathFunctions::Create("2*.2*(1 + sin(2*_pi*(x-4*t)/.01))/2, 2*.3*(1 + sin(2*_pi*(x-4*t)/.01))/2, 2*(1-.5*(1 + sin(2*_pi*(x-4*t)/.01))/2)"),
                                 .expectedL2Convergence = {NAN, NAN, NAN, NAN, 1, 1, 1},
                                 .expectedLInfConvergence = {NAN, NAN, NAN, NAN, 1, 1, 1}}),
                         [](const testing::TestParamInfo<CompressibleFlowAdvectionTestParameters> &info) { return info.param.mpiTestParameter.getTestName(); });
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
