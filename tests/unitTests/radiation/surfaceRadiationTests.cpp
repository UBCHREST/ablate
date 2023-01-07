#include <petsc.h>
#include <petsc/private/dmpleximpl.h>
#include <mathFunctions/functionFactory.hpp>
#include <memory>
#include "MpiTestFixture.hpp"
#include "builder.hpp"
#include "convergenceTester.hpp"
#include "domain/boxMeshBoundaryCells.hpp"
#include "domain/modifiers/ghostBoundaryCells.hpp"
#include "environment/runEnvironment.hpp"
#include "eos/perfectGas.hpp"
#include "eos/radiationProperties/constant.hpp"
#include "eos/radiationProperties/radiationProperties.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "gtest/gtest.h"
#include "monitors/timeStepMonitor.hpp"
#include "parameters/mapParameters.hpp"
#include "radiation/radiation.hpp"
#include "radiation/raySharingRadiation.hpp"
#include "radiation/surfaceRadiation.hpp"
#include "radiation/volumeRadiation.hpp"
#include "utilities/petscUtilities.hpp"

struct RadiationTestParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    std::vector<int> meshFaces;
    std::vector<double> meshStart;
    std::vector<double> meshEnd;
    std::function<std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>>()> initialization;
    std::shared_ptr<ablate::mathFunctions::MathFunction> expectedResult;
    std::function<std::shared_ptr<ablate::radiation::Radiation>(std::shared_ptr<ablate::eos::radiationProperties::RadiationModel> radiationModelIn)> radiationFactory;
    std::string emitLabel;
    const char* detectLabel;
};

class RadiationTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<RadiationTestParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

static PetscReal ComputeParallelViewFactor() {
    /** Computes the analytical solution for the view factor between two parallel plates based on the geometric dimensions
     * */
}

static PetscReal ComputePerpendicularViewFactor() {
    /** Computes the analytical solution for the view factor between two perpendicular plates based on the geometric dimensions
     * */
}

TEST_P(RadiationTestFixture, ShouldComputeCorrectSourceTerm) {
    StartWithMPI

        // initialize petsc and mpi
        ablate::environment::RunEnvironment::Initialize(argc, argv);
        ablate::utilities::PetscUtilities::Initialize();

        //! Create regions for the test
        auto emitRegion = std::make_shared<ablate::domain::Region>(GetParam().emitLabel);
        auto detectRegion = std::make_shared<ablate::domain::Region>(GetParam().detectLabel);

        // keep track of history
        testingResources::ConvergenceTester l2History("l2");

        auto eos = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}}));

        // determine required fields for radiation, this will include euler and temperature
        std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptors = {std::make_shared<ablate::finiteVolume::CompressibleFlowFields>(eos)};

        auto domain = std::make_shared<ablate::domain::BoxMeshBoundaryCells>("simpleMesh",
                                                                             fieldDescriptors,
                                                                             std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{},
                                                                             std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{},
                                                                             GetParam().meshFaces,
                                                                             GetParam().meshStart,
                                                                             GetParam().meshEnd,
                                                                             false,
                                                                             ablate::parameters::MapParameters::Create({{"dm_plex_hash_location", "true"}}));

        DMView(domain->GetDM(), PETSC_VIEWER_STDOUT_WORLD);

        // Setup the flow data
        auto parameters = std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"cfl", ".4"}});

        // Set the initial conditions for euler (not used, so set all to zero)
        auto initialConditionEuler = std::make_shared<ablate::mathFunctions::FieldFunction>("euler", std::make_shared<ablate::mathFunctions::ConstantValue>(0.0));

        // create a time stepper
        auto timeStepper = ablate::solver::TimeStepper("timeStepper", domain, ablate::parameters::MapParameters::Create({{"ts_max_steps", "0"}}), {}, {initialConditionEuler});

        // Create an instance of radiation
        auto radiationPropertiesModel = std::make_shared<ablate::eos::radiationProperties::Constant>(0.0);  //! A transparent domain will enable a surface exchange solution.
        auto radiationModel =
            GetParam().radiationFactory(radiationPropertiesModel);  //! This is the surface radiation solver which must be slightly modified in order to produce the view factor problem.
        auto interiorLabel =
            std::make_shared<ablate::domain::Region>("interiorCells");  //! Use only the interior cell region for the region of the radiation solver in order to pass boundary conditions
        //        auto radiation = std::make_shared<ablate::radiation::VolumeRadiation>("radiation", interiorLabel, nullptr, radiationModel, nullptr, nullptr);

        //! The time stepping procedure is not necessary. The surface radiation class is not a solver and does not need to complete time steps in order to produce a solution.
        // register the flowSolver with the timeStepper.
        //        timeStepper.Register(radiation, {std::make_shared<ablate::monitors::TimeStepMonitor>()});
        timeStepper.Solve();

        // force the aux variables of temperature to a known value
        auto auxVec = domain->GetSubDomain(ablate::domain::Region::ENTIREDOMAIN)->GetAuxVector();
        domain->GetSubDomain(ablate::domain::Region::ENTIREDOMAIN)->ProjectFieldFunctionsToLocalVector(GetParam().initialization(), auxVec);

        // Setup the rhs for the test
        Vec rhs;
        DMGetLocalVector(domain->GetDM(), &rhs) >> testErrorChecker;
        VecZeroEntries(rhs) >> testErrorChecker;

        // Apply the rhs function for the radiation solver
        //        radiationModel->PreRHSFunction(timeStepper.GetTS(), 0.0, true, nullptr) >> testErrorChecker;
        //        radiationModel->ComputeRHSFunction(0, rhs, rhs);  // The ray tracing function needs to be renamed in order to occupy the role of compute right hand side function

        // determine the euler field
        const auto& eulerFieldInfo = domain->GetField("euler");

        // Set up the surface radiation solver
        // check for ghost cells
        DMLabel ghostLabel;
        DMGetLabel(domain->GetSubDomain(interiorLabel)->GetDM(), "ghost", &ghostLabel) >> testErrorChecker;

        DMLabel detectLabel;
        DMGetLabel(domain->GetSubDomain(ablate::domain::Region::ENTIREDOMAIN)->GetDM(), GetParam().detectLabel, &detectLabel);

        /** Get the face range of the entire mesh so that the faces with the correct label can be isolated out of it
         * */
        ablate::solver::Range meshFaceRange;
        PetscInt depth;
        DMPlexGetDepth(domain->GetSubDomain(ablate::domain::Region::ENTIREDOMAIN)->GetDM(), &depth) >> testErrorChecker;
        /** Get the range of the  */
        {
            IS allPointIS;
            DMGetStratumIS(domain->GetSubDomain(ablate::domain::Region::ENTIREDOMAIN)->GetDM(), "dim", depth, &allPointIS) >> testErrorChecker;
            if (!allPointIS) {
                DMGetStratumIS(domain->GetSubDomain(ablate::domain::Region::ENTIREDOMAIN)->GetDM(), "depth", depth, &allPointIS) >> testErrorChecker;
            }

            // If there is a label for this solver, get only the parts of the mesh that here
            if (detectLabel) {
                DMLabel label;
                DMGetLabel(domain->GetSubDomain(ablate::domain::Region::ENTIREDOMAIN)->GetDM(), detectRegion->GetName().c_str(), &label);

                IS labelIS;
                DMLabelGetStratumIS(label, detectRegion->GetValue(), &labelIS) >> testErrorChecker;
                ISIntersect_Caching_Internal(allPointIS, labelIS, &meshFaceRange.is) >> testErrorChecker;
                ISDestroy(&labelIS) >> testErrorChecker;
            } else {
                PetscObjectReference((PetscObject)allPointIS) >> testErrorChecker;
                meshFaceRange.is = allPointIS;
            }

            // Get the point range
            if (meshFaceRange.is == nullptr) {
                // There are no points in this region, so skip
                meshFaceRange.start = 0;
                meshFaceRange.end = 0;
                meshFaceRange.points = nullptr;
            } else {
                // Get the range
                ISGetPointRange(meshFaceRange.is, &meshFaceRange.start, &meshFaceRange.end, &meshFaceRange.points) >> testErrorChecker;
            }

            // Clean up the allCellIS
            ISDestroy(&allPointIS) >> testErrorChecker;
        }
        //!< Get the face range of the boundary cells to initialize the rays with this range. Add all of the faces to this range that belong to the boundary solver.
        ablate::solver::DynamicRange faceRange;
        for (PetscInt c = meshFaceRange.start; c < meshFaceRange.end; ++c) {
            const PetscInt iFace = meshFaceRange.points ? meshFaceRange.points[c] : c;  //!< Isolates the valid cells
            PetscInt ghost = -1;
            PetscInt detect = -1;
            if (ghostLabel) DMLabelGetValue(ghostLabel, iFace, &ghost) >> testErrorChecker;
            if (detectLabel) DMLabelGetValue(detectLabel, iFace, &detect) >> testErrorChecker;
            if ((ghost < 0) && (detect >= 1)) faceRange.Add(iFace);  //!< Add each ID to the range that the radiation solver will use
        }
        radiationModel->Setup(faceRange.GetRange(), *(domain->GetSubDomain(interiorLabel)));
        radiationModel->Initialize(faceRange.GetRange(), *(domain->GetSubDomain(interiorLabel)));  //!< Pass the non-dynamic range into the radiation solver.

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

            /// Declare L2 norm variables
            PetscReal radsum = 0.0;
            // TODO: The radiation must be summed for the faces in the irradiated region.

            //  March over each stored value in the surface radiation result and sum them to determine the total irradiation to that surface.
            for (PetscInt c = meshFaceRange.start; c < meshFaceRange.end; ++c) {
                const PetscInt iFace = meshFaceRange.points ? meshFaceRange.points[c] : c;

                if (ablate::domain::Region::InRegion(detectRegion, dmCell, iFace)) {
                    // Get the cell center
                    PetscFVCellGeom* cellGeom;
                    DMPlexPointLocalRead(dmCell, iFace, cellGeomArray, &cellGeom) >> testErrorChecker;

                    // extract the result from the stored solver value
                    /// Summing of the irradiation
                    radsum += radiationModel->origin[iFace].intensity;
                }
            }
            /// Compute the L2 Norm error
            //            double N = (cellRange.end - cellRange.start);
            //            double l2 = sqrt(l2sum) / N;

            //            PetscPrintf(MPI_COMM_WORLD, "L2 Norm: %f\n", sqrt(l2sum) / N);
            //! Compute the difference between the analytical and computational solutions
            PetscReal error = 0;

            if (error > 1) {
                FAIL() << "Radiation test error exceeded.";
            }

            VecRestoreArrayRead(rhs, &rhsArray) >> testErrorChecker;
            VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> testErrorChecker;
        }

        DMViewFromOptions(domain->GetSubDomain(ablate::domain::Region::ENTIREDOMAIN)->GetAuxDM(), nullptr, "-viewdm");
        VecViewFromOptions(auxVec, nullptr, "-viewvec");

        // TODO: Return the face ranges and other cleanup that might be necessary

        DMRestoreLocalVector(domain->GetDM(), &rhs);
        ablate::environment::RunEnvironment::Finalize();
        exit(0);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    RadiationTests, RadiationTestFixture,
    testing::Values(
        (RadiationTestParameters){.mpiTestParameter = {.testName = "Parallel Plates 1 proc.", .nproc = 1},
                                  .meshFaces = {20, 20, 20},
                                  .meshStart = {0, 0, 0},
                                  .meshEnd = {1, 1, 1},
                                  .initialization =
                                      []() {
                                          return std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>>{
                                              std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                     ablate::mathFunctions::Create("y < 0 ? (-6.349E6*y*y + 2000.0) : (-1.179E7*y*y + 2000.0)")),
                                              std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                     ablate::mathFunctions::Create("1300"),
                                                                                                     nullptr,
                                                                                                     std::make_shared<ablate::domain::Region>("boundaryCellsBottom")),
                                              std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                     ablate::mathFunctions::Create("700"),
                                                                                                     nullptr,
                                                                                                     std::make_shared<ablate::domain::Region>("boundaryCellsTop"))};
                                      },
                                  .expectedResult = ablate::mathFunctions::Create("x + y"),
                                  .radiationFactory =
                                      [](std::shared_ptr<ablate::eos::radiationProperties::RadiationModel> radiationModelIn) {
                                          auto interiorLabel = std::make_shared<ablate::domain::Region>("interiorCells");
                                          return std::make_shared<ablate::radiation::SurfaceRadiation>("radiationBase", interiorLabel, 15, radiationModelIn, nullptr);
                                      },
                                  .emitLabel = "boundaryCellsLeft",      //! Label of the region from which the radiation is emitted
                                  .detectLabel = "boundaryCellsRight"},  //! Label of the region where the radiation is detected by the surface solver
        (RadiationTestParameters){.mpiTestParameter = {.testName = "Perpendicular Plates 1 proc.", .nproc = 1},
                                  .meshFaces = {20, 20, 20},
                                  .meshStart = {0, 0, 0},
                                  .meshEnd = {1, 1, 1},
                                  .initialization =
                                      []() {
                                          return std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>>{
                                              std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                     ablate::mathFunctions::Create("y < 0 ? (-6.349E6*y*y + 2000.0) : (-1.179E7*y*y + 2000.0)")),
                                              std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                     ablate::mathFunctions::Create("1300"),
                                                                                                     nullptr,
                                                                                                     std::make_shared<ablate::domain::Region>("boundaryCellsBottom")),
                                              std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                     ablate::mathFunctions::Create("700"),
                                                                                                     nullptr,
                                                                                                     std::make_shared<ablate::domain::Region>("boundaryCellsTop"))};
                                      },
                                  .expectedResult = ablate::mathFunctions::Create("x + y"),
                                  .radiationFactory =
                                      [](std::shared_ptr<ablate::eos::radiationProperties::RadiationModel> radiationModelIn) {
                                          auto interiorLabel = std::make_shared<ablate::domain::Region>("interiorCells");
                                          return std::make_shared<ablate::radiation::SurfaceRadiation>("radiationBase", interiorLabel, 15, radiationModelIn, nullptr);
                                      },
                                  .emitLabel = "boundaryCellsLeft",      //! Label of the region from which the radiation is emitted
                                  .detectLabel = "boundaryCellsRight"},  //! Label of the region where the radiation is detected by the surface solver
        (RadiationTestParameters){.mpiTestParameter = {.testName = "Parallel Plates 2 proc.", .nproc = 2},
                                  .meshFaces = {20, 20, 20},
                                  .meshStart = {0, 0, 0},
                                  .meshEnd = {1, 1, 1},
                                  .initialization =
                                      []() {
                                          return std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>>{
                                              std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                     ablate::mathFunctions::Create("y < 0 ? (-6.349E6*y*y + 2000.0) : (-1.179E7*y*y + 2000.0)")),
                                              std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                     ablate::mathFunctions::Create("1300"),
                                                                                                     nullptr,
                                                                                                     std::make_shared<ablate::domain::Region>("boundaryCellsBottom")),
                                              std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                     ablate::mathFunctions::Create("700"),
                                                                                                     nullptr,
                                                                                                     std::make_shared<ablate::domain::Region>("boundaryCellsTop"))};
                                      },
                                  .expectedResult = ablate::mathFunctions::Create("x + y"),
                                  .radiationFactory =
                                      [](std::shared_ptr<ablate::eos::radiationProperties::RadiationModel> radiationModelIn) {
                                          auto interiorLabel = std::make_shared<ablate::domain::Region>("interiorCells");
                                          return std::make_shared<ablate::radiation::SurfaceRadiation>("radiationBase", interiorLabel, 15, radiationModelIn, nullptr);
                                      },
                                  .emitLabel = "boundaryCellsLeft",      //! Label of the region from which the radiation is emitted
                                  .detectLabel = "boundaryCellsRight"},  //! Label of the region where the radiation is detected by the surface solver
        (RadiationTestParameters){.mpiTestParameter = {.testName = "Perpendicular Plates 2 proc.", .nproc = 2},
                                  .meshFaces = {20, 20, 20},
                                  .meshStart = {0, 0, 0},
                                  .meshEnd = {1, 1, 1},
                                  .initialization =
                                      []() {
                                          return std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>>{
                                              std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                     ablate::mathFunctions::Create("y < 0 ? (-6.349E6*y*y + 2000.0) : (-1.179E7*y*y + 2000.0)")),
                                              std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                     ablate::mathFunctions::Create("1300"),
                                                                                                     nullptr,
                                                                                                     std::make_shared<ablate::domain::Region>("boundaryCellsBottom")),
                                              std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                     ablate::mathFunctions::Create("700"),
                                                                                                     nullptr,
                                                                                                     std::make_shared<ablate::domain::Region>("boundaryCellsTop"))};
                                      },
                                  .expectedResult = ablate::mathFunctions::Create("x + y"),
                                  .radiationFactory =
                                      [](std::shared_ptr<ablate::eos::radiationProperties::RadiationModel> radiationModelIn) {
                                          return std::make_shared<ablate::radiation::SurfaceRadiation>("radiationBase", ablate::domain::Region::ENTIREDOMAIN, 15, radiationModelIn, nullptr);
                                      },
                                  .emitLabel = "boundaryCellsLeft",       //! Label of the region from which the radiation is emitted
                                  .detectLabel = "boundaryCellsRight"}),  //! Label of the region where the radiation is detected by the surface solver
    [](const testing::TestParamInfo<RadiationTestParameters>& info) { return info.param.mpiTestParameter.getTestName(); });