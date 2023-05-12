#include <petsc.h>
#include <memory>
#include <vector>
#include "MpiTestFixture.hpp"
#include "domain/boxMesh.hpp"
#include "domain/modifiers/ghostBoundaryCells.hpp"
#include "environment/runEnvironment.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "finiteVolume/processes/surfaceForce.hpp"
#include "finiteVolume/processes/twoPhaseEulerAdvection.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"

struct SurfaceForceTestParameters {
    PetscInt dim;
    PetscInt cellNumber;
    std::vector<int> meshFaces;
    std::vector<double> meshStart;
    std::vector<double> meshEnd;
    std::vector<PetscReal> inputEulerValues;
    std::shared_ptr<ablate::mathFunctions::MathFunction> inputVFfield;
    std::vector<PetscReal> expectedEulerSource;
};

class SurfaceForceTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<SurfaceForceTestParameters> {};

TEST_P(SurfaceForceTestFixture, ShouldComputeCorrectSurfaceForce) {
    StartWithMPI
        {
            ablate::environment::RunEnvironment::Initialize(argc, argv);
            ablate::utilities::PetscUtilities::Initialize();
            PetscReal errorTolerance = 1E-3;
            PetscReal sigma = 0.07;

            auto eos = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}}));

            // define a test fields
            std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptors = {
                std::make_shared<ablate::finiteVolume::CompressibleFlowFields>(eos),

                std::make_shared<ablate::domain::FieldDescription>(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD,
                                                                   "",
                                                                   ablate::domain::FieldDescription::ONECOMPONENT,
                                                                   ablate::domain::FieldLocation::SOL,
                                                                   ablate::domain::FieldType::FVM)};

            auto dim = GetParam().dim;
            // define the test mesh
            auto domain =
                std::make_shared<ablate::domain::BoxMesh>("test",

                                                          fieldDescriptors,
                                                          std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<ablate::domain::modifiers::GhostBoundaryCells>()},

                                                          GetParam().meshFaces,
                                                          GetParam().meshStart,
                                                          GetParam().meshEnd,
                                                          std::vector<std::string>(dim, "NONE") /*boundary*/,
                                                          false /*simplex*/

                );
            DMCreateLabel(domain->GetDM(), "ghost");
            auto initialConditionFV = std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD, GetParam().inputVFfield);
            auto initialConditionEuler = std::make_shared<ablate::mathFunctions::FieldFunction>("euler", std::make_shared<ablate::mathFunctions::ConstantValue>(1));
            // the solver
            auto fvSolver = std::make_shared<ablate::finiteVolume::FiniteVolumeSolver>("testSolver",
                                                                                       ablate::domain::Region::ENTIREDOMAIN,
                                                                                       nullptr,
                                                                                       std::vector<std::shared_ptr<ablate::finiteVolume::processes::Process>>(),
                                                                                       std::vector<std::shared_ptr<ablate::finiteVolume::boundaryConditions::BoundaryCondition>>{});
            // initialize it
            domain->InitializeSubDomains({fvSolver}, {initialConditionFV, initialConditionEuler});
            PetscScalar *eulerSource = nullptr;
            Vec computedF;
            PetscScalar *sourceArray;
            PetscScalar *solution;
            Vec locSolution;
            DMGetLocalVector(domain->GetDM(), &locSolution);
            DMGlobalToLocal(domain->GetDM(), domain->GetSolutionVector(), INSERT_VALUES, locSolution);
            VecGetArray(locSolution, &solution);

            // copy over euler
            PetscScalar *eulerField = nullptr;
            DMPlexPointLocalFieldRef(domain->GetDM(), GetParam().cellNumber, domain->GetField("euler").id, solution, &eulerField);
            // copy over euler
            for (std::size_t i = 0; i < GetParam().inputEulerValues.size(); i++) {
                eulerField[i] = GetParam().inputEulerValues[i];
            }

            VecRestoreArray(locSolution, &solution);

            auto process = ablate::finiteVolume::processes::SurfaceForce(sigma);
            process.Setup(*fvSolver);
            DMGetLocalVector(domain->GetDM(), &computedF);
            VecZeroEntries(computedF);

            ablate::finiteVolume::processes::SurfaceForce::ComputeSource(*fvSolver, domain->GetDM(), 0.0, locSolution, computedF, &process);

            // ASSERT
            VecGetArray(computedF, &sourceArray);

            DMPlexPointLocalFieldRef(domain->GetDM(), GetParam().cellNumber, domain->GetField("euler").id, sourceArray, &eulerSource);
            for (std::size_t c = 0; c < GetParam().expectedEulerSource.size(); c++) {
                ASSERT_LT(PetscAbs((GetParam().expectedEulerSource[c] - eulerSource[c]) / (GetParam().expectedEulerSource[c] + 1E-30)), errorTolerance)
                    << "The percent difference for the expected and actual source (" << GetParam().expectedEulerSource[c] << " vs " << eulerSource[c] << ") should be small for index " << c;
            }
            VecRestoreArray(computedF, &sourceArray) >> ablate::utilities::PetscUtilities::checkError;
            DMRestoreLocalVector(domain->GetDM(), &locSolution) >> ablate::utilities::PetscUtilities::checkError;
            DMRestoreLocalVector(domain->GetDM(), &computedF) >> ablate::utilities::PetscUtilities::checkError;
        }
        ablate::environment::RunEnvironment::Finalize();
        exit(0);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(SurfaceForce, SurfaceForceTestFixture,
                         testing::Values((SurfaceForceTestParameters){.dim = 1,
                                                                      .cellNumber = 1,
                                                                      .meshFaces = {3},
                                                                      .meshStart = {0},
                                                                      .meshEnd = {1},
                                                                      .inputEulerValues = {1, 0, 0},
                                                                      .inputVFfield = ablate::mathFunctions::Create(" x<2/3 ? 1:0"),
                                                                      .expectedEulerSource = {0, 0, 0}},
                                         (SurfaceForceTestParameters){.dim = 2,
                                                                      .cellNumber = 4,
                                                                      .meshFaces = {3, 3},
                                                                      .meshStart = {0, 0},
                                                                      .meshEnd = {1, 1},
                                                                      .inputEulerValues = {1, 0, 0, 0},
                                                                      .inputVFfield = ablate::mathFunctions::Create("1"),
                                                                      .expectedEulerSource = {0, 0, 0, 0}},
                                         (SurfaceForceTestParameters){.dim = 2,
                                                                      .cellNumber = 4,
                                                                      .meshFaces = {3, 3},
                                                                      .meshStart = {0, 0},
                                                                      .meshEnd = {1, 1},
                                                                      .inputEulerValues = {1, 0, 0, 0},
                                                                      .inputVFfield = ablate::mathFunctions::Create(" x<2/3 && y< 2/3 ? 1:0"),
                                                                      .expectedEulerSource = {0, 0, -0.445477, -0.445477}},
                                         (SurfaceForceTestParameters){.dim = 3,
                                                                      .cellNumber = 13,
                                                                      .meshFaces = {3, 3, 3},
                                                                      .meshStart = {0, 0, 0},
                                                                      .meshEnd = {1, 1, 1},
                                                                      .inputEulerValues = {1, 0, 0, 0, 0},
                                                                      .inputVFfield = ablate::mathFunctions::Create(" x<2/3 && y< 2/3  && z< 1? 1:0"),
                                                                      .expectedEulerSource = {0, 0, -0.445477, -0.445477, 0}},  // should be getting same results as 2D
                                         (SurfaceForceTestParameters){.dim = 3,
                                                                      .cellNumber = 13,
                                                                      .meshFaces = {3, 3, 3},
                                                                      .meshStart = {0, 0, 0},
                                                                      .meshEnd = {1, 1, 1},
                                                                      .inputEulerValues = {1, 0, 1, 1, 0},
                                                                      .inputVFfield = ablate::mathFunctions::Create(" x<2/3 && y< 2/3  && z< 1? 1:0"),
                                                                      .expectedEulerSource = {0, -0.890954544, -0.445477, -0.445477, 0}}  // should calculate energy also

                                         ),
                         [](const testing::TestParamInfo<SurfaceForceTestParameters> &info) { return "SurfaceForceTest" + std::to_string(info.index); });