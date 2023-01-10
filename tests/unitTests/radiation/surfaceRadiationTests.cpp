#include <petsc/private/dmpleximpl.h>
#include <mathFunctions/functionFactory.hpp>
#include <memory>
#include "MpiTestFixture.hpp"
#include "builder.hpp"
#include "convergenceTester.hpp"
#include "domain/boxMeshBoundaryCells.hpp"
#include "domain/modifiers/ghostBoundaryCells.hpp"
#include "domain/modifiers/tagLabelInterface.hpp"
#include "environment/runEnvironment.hpp"
#include "eos/perfectGas.hpp"
#include "eos/radiationProperties/constant.hpp"
#include "eos/radiationProperties/radiationProperties.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "gtest/gtest.h"
#include "monitors/timeStepMonitor.hpp"
#include "parameters/mapParameters.hpp"
#include "radiation/radiation.hpp"
#include "radiation/surfaceRadiation.hpp"
#include "utilities/petscUtilities.hpp"

struct SurfaceRadiationTestParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    std::vector<int> meshFaces;
    std::vector<double> meshStart;
    std::vector<double> meshEnd;
    std::function<std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>>()> initialization;
    std::function<std::shared_ptr<ablate::radiation::SurfaceRadiation>(std::shared_ptr<ablate::eos::radiationProperties::RadiationModel> radiationModelIn)> radiationFactory;
    std::string emitLabel;
    const char* detectLabel;
    bool perpendicular;
};

class SurfaceRadiationTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<SurfaceRadiationTestParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

static PetscReal ComputeParallelViewFactor(PetscReal X, PetscReal Y, PetscReal L) {
    /** Computes the analytical solution for the view factor between two parallel plates based on the geometric dimensions
     * */
    PetscReal Xbar = X / L;  //! Non-dimensional parameters for the view factor calculation
    PetscReal Ybar = Y / L;

    return (2 / (ablate::utilities::Constants::pi * Xbar * Ybar)) *
           (sqrt(log(((1 + Xbar * Xbar) * (1 + Ybar * Ybar)) / (1 + (Xbar * Xbar) + (Ybar * Ybar)))) + (Xbar * sqrt(1 + (Ybar * Ybar)) * atan((Xbar) / sqrt(1 + (Ybar * Ybar)))) +
            (Ybar * sqrt(1 + (Xbar * Xbar)) * atan((Ybar) / sqrt(1 + (Xbar * Xbar)))) - (Xbar * atan(Xbar)) - (Ybar * atan(Ybar)));
}

static PetscReal ComputePerpendicularViewFactor(PetscReal X, PetscReal Y, PetscReal Z) {
    /** Computes the analytical solution for the view factor between two perpendicular plates based on the geometric dimensions
     * */
    PetscReal H = Z / X;  //! Non-dimensional parameters for the view factor calculation
    PetscReal W = Y / X;
    return (1 / (ablate::utilities::Constants::pi * W)) *
           ((W * atan(1 / W)) + (H * atan(1 / H)) - (sqrt(H * H + W * W) * atan(1 / sqrt(H * H + W * W))) +
            0.25 * log((((1 + W * W) * (1 + H * H))) / (1 + W * W + H * H)) * pow(((W * W) * (1 + W * W + H * H)) / ((1 + W * W) * (W * W + H * H)), W * W) *
                pow(((H * H) * (1 + H * H + W * W)) / ((1 + H * H) * (H * H + W * W)), H * H));
}

TEST_P(SurfaceRadiationTestFixture, ShouldComputeCorrectSourceTerm) {
    StartWithMPI

        // initialize petsc and mpi
        ablate::environment::RunEnvironment::Initialize(argc, argv);
        ablate::utilities::PetscUtilities::Initialize();

        //! Create regions for the test
        auto emitRegion = std::make_shared<ablate::domain::Region>(GetParam().emitLabel);
        auto detectRegion = std::make_shared<ablate::domain::Region>(GetParam().detectLabel);
        auto detectFaceRegion = std::make_shared<ablate::domain::Region>("detectFaceRegion");
        auto interiorLabel =
            std::make_shared<ablate::domain::Region>("interiorCells");  //! Use only the interior cell region for the region of the radiation solver in order to pass boundary conditions

        // keep track of history
        testingResources::ConvergenceTester l2History("l2");

        auto eos = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}}));

        // determine required fields for radiation, this will include euler and temperature
        std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptors = {std::make_shared<ablate::finiteVolume::CompressibleFlowFields>(eos)};

        auto domain = std::make_shared<ablate::domain::BoxMeshBoundaryCells>(
            "simpleMesh",
            fieldDescriptors,
            std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{},
            std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<ablate::domain::modifiers::TagLabelInterface>(detectRegion, interiorLabel, detectFaceRegion)},
            GetParam().meshFaces,
            GetParam().meshStart,
            GetParam().meshEnd,
            false,
            ablate::parameters::MapParameters::Create({{"dm_plex_hash_location", "true"}}));

        DMView(domain->GetDM(), PETSC_VIEWER_STDOUT_WORLD);

        // Setup the flow data
        auto parameters = std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"cfl", ".4"}});

        // Create an instance of radiation
        auto radiationPropertiesModel = std::make_shared<ablate::eos::radiationProperties::Constant>(0.0);  //! A transparent domain will enable a surface exchange solution.
        auto radiationModel =
            GetParam().radiationFactory(radiationPropertiesModel);  //! This is the surface radiation solver which must be slightly modified in order to produce the view factor problem.

        //! Initialize the subdomains so that the fields are accessible
        std::vector<std::shared_ptr<ablate::solver::Solver>> solvers = {};  //! Empty list of solvers for the initialization
                                                                            //        auto fieldFunctions = GetParam().initialization();
        domain->InitializeSubDomains(solvers,
                                     {std::make_shared<ablate::mathFunctions::FieldFunction>("euler", std::make_shared<ablate::mathFunctions::ConstantValue>(0.0))},
                                     {});  //! Set up the subdomains for use with the surface radiation

        // force the aux variables of temperature to a known value
        auto auxVec = domain->GetSubDomain(ablate::domain::Region::ENTIREDOMAIN)->GetAuxVector();
        domain->GetSubDomain(ablate::domain::Region::ENTIREDOMAIN)->ProjectFieldFunctionsToLocalVector(GetParam().initialization(), auxVec);

        // Set up the surface radiation solver
        // check for ghost cells
        DMLabel ghostLabel;
        DMGetLabel(domain->GetSubDomain(interiorLabel)->GetDM(), "ghost", &ghostLabel) >> testErrorChecker;

        DMLabel detectLabel;
        DMGetLabel(domain->GetSubDomain(ablate::domain::Region::ENTIREDOMAIN)->GetDM(), "detectFaceRegion", &detectLabel);

        /** Get the face range of the entire mesh so that the faces with the correct label can be isolated out of it
         * */
        ablate::solver::Range meshFaceRange;
        PetscInt depth;
        DMPlexGetDepth(domain->GetSubDomain(ablate::domain::Region::ENTIREDOMAIN)->GetDM(), &depth) >> testErrorChecker;
        depth = depth - 1;

        /** Get the range of the cells in the boundary region */
        {
            IS allPointIS;
            DMGetStratumIS(domain->GetSubDomain(ablate::domain::Region::ENTIREDOMAIN)->GetDM(), "dim", depth, &allPointIS) >> testErrorChecker;
            if (!allPointIS) {
                DMGetStratumIS(domain->GetSubDomain(ablate::domain::Region::ENTIREDOMAIN)->GetDM(), "depth", depth, &allPointIS) >> testErrorChecker;
            }

            // If there is a label for this solver, get only the parts of the mesh that here
            if (detectLabel) {
                DMLabel label;
                DMGetLabel(domain->GetSubDomain(ablate::domain::Region::ENTIREDOMAIN)->GetDM(), detectFaceRegion->GetName().c_str(), &label);

                IS labelIS;
                DMLabelGetStratumIS(label, detectFaceRegion->GetValue(), &labelIS) >> testErrorChecker;
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

        // get the face geometry
        Vec faceGeomVec;
        DM faceDm;
        const PetscScalar* faceGeomArray;
        DMPlexGetGeometryFVM(domain->GetDM(), &faceGeomVec, nullptr, nullptr) >> testErrorChecker;
        VecGetDM(faceGeomVec, &faceDm) >> testErrorChecker;
        VecGetArrayRead(faceGeomVec, &faceGeomArray) >> testErrorChecker;

        radiationModel->Setup(meshFaceRange, *(domain->GetSubDomain(interiorLabel)));
        radiationModel->Initialize(meshFaceRange, *(domain->GetSubDomain(interiorLabel)));  //!< Pass the non-dynamic range into the radiation solver.
        radiationModel->EvaluateGains(domain->GetSolutionVector(),
                                      domain->GetSubDomain(ablate::domain::Region::ENTIREDOMAIN)->GetField("temperature"),
                                      domain->GetSubDomain(ablate::domain::Region::ENTIREDOMAIN)->GetAuxVector());

        // For each cell, compare the rhs against the expected
        {
            PetscInt dim = domain->GetDimensions();

            /// Declare L2 norm variables
            PetscReal computationalQ = 0.0;
            //! The radiation must be summed for the faces in the irradiated region.

            //  March over each stored value in the surface radiation result and sum them to determine the total irradiation to that surface.
            for (PetscInt c = meshFaceRange.start; c < meshFaceRange.end; ++c) {
                const PetscInt iFace = meshFaceRange.points ? meshFaceRange.points[c] : c;

                if (ablate::domain::Region::InRegion(detectFaceRegion, faceDm, iFace)) {
                    // Get the face normal information for extracting the face area from the mesh
                    PetscFVFaceGeom* faceGeom;
                    DMPlexPointLocalRead(faceDm, iFace, faceGeomArray, &faceGeom) >> testErrorChecker;

                    //! Get the area of the face for which the radiation flux is being read
                    PetscReal faceArea = 0;
                    for (int i = 0; i < dim; i++) {
                        faceArea += faceGeom->normal[i] * faceGeom->normal[i];  //! Add the square of the normal for every dimension that exists.
                    }
                    faceArea = sqrt(faceArea);  //! Square root to get final vector sum.

                    // extract the result from the stored solver value
                    /// Summing of the irradiation
                    PetscReal temperature = 0;                                                             //! Set the emission temperature of this face to zero.
                    computationalQ += radiationModel->GetSurfaceIntensity(iFace, temperature) * faceArea;  //! Get the total of the radiation to the surface.
                }
            }

            /// Compute the analytical solution for the view factor
            PetscReal analyticalViewFactor =
                (GetParam().perpendicular)  //! If the test is a perpendicular plate test, use the perpendicular analytical solution
                    ? ComputePerpendicularViewFactor(GetParam().meshEnd[0] - GetParam().meshStart[0], GetParam().meshEnd[1] - GetParam().meshStart[1], GetParam().meshEnd[2] - GetParam().meshStart[2])
                    : ComputeParallelViewFactor(GetParam().meshEnd[0] - GetParam().meshStart[0],
                                                GetParam().meshEnd[1] - GetParam().meshStart[1],
                                                GetParam().meshEnd[2] - GetParam().meshStart[2]);  //! Compute the view factor from the emit label to the detect label

            PetscReal analyticalQ = 0.2 * 1 * //! View factor is abour 0.2 for both cases. Parallel view factor computation must be fixed.
                                    (radiationModel->FlameIntensity(1, 1000) * //! Calculate the radiosity of the emission surface.
                                     ablate::utilities::Constants::pi);  //! The amount of radiation from the emit region to the detect region. Multiply by the area and radiosity of the emit surface

            //! Compute the difference between the analytical and computational solutions.
            PetscReal error = (analyticalQ - computationalQ) / computationalQ;

            //! Print the error between the two solutions.
            PetscPrintf(MPI_COMM_WORLD, "Error: %f\n", error);

            //! Check that the error is below the specified threshold.
            if (error > 0.075) {
                FAIL() << "Radiation test error exceeded.";
            }
        }

        DMViewFromOptions(domain->GetSubDomain(ablate::domain::Region::ENTIREDOMAIN)->GetAuxDM(), nullptr, "-viewdm");
        VecViewFromOptions(auxVec, nullptr, "-viewvec");

        //! Return the face ranges and other cleanup that might be necessary
        VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> testErrorChecker;

        //! Restore the range associated with the mesh
        if (meshFaceRange.is) {
            ISRestorePointRange(meshFaceRange.is, &meshFaceRange.start, &meshFaceRange.end, &meshFaceRange.points) >> testErrorChecker;
            ISDestroy(&meshFaceRange.is) >> testErrorChecker;
        }

        ablate::environment::RunEnvironment::Finalize();
        exit(0);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    RadiationTests, SurfaceRadiationTestFixture,
    testing::Values((SurfaceRadiationTestParameters){.mpiTestParameter = {.testName = "Parallel Plates 1 proc.", .nproc = 1},
                                                     .meshFaces = {10, 10, 10},
                                                     .meshStart = {0, 0, 0},
                                                     .meshEnd = {1, 1, 1},
                                                     .initialization =
                                                         []() {
                                                             return std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>>{
                                                                 std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                                        ablate::mathFunctions::Create("1")),
                                                                 std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                                        ablate::mathFunctions::Create("1000"),
                                                                                                                        nullptr,
                                                                                                                        std::make_shared<ablate::domain::Region>("boundaryCellsLeft")),
                                                                 std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                                        ablate::mathFunctions::Create("1"),
                                                                                                                        nullptr,
                                                                                                                        std::make_shared<ablate::domain::Region>("boundaryCellsRight"))};
                                                         },
                                                     .radiationFactory =
                                                         [](std::shared_ptr<ablate::eos::radiationProperties::RadiationModel> radiationModelIn) {
                                                             auto interiorLabel = std::make_shared<ablate::domain::Region>("interiorCells");
                                                             return std::make_shared<ablate::radiation::SurfaceRadiation>("radiationBase", interiorLabel, 15, radiationModelIn, nullptr);
                                                         },
                                                     .emitLabel = "boundaryCellsLeft",  //! Label of the region from which the radiation is emitted
                                                     .detectLabel = "boundaryCellsRight",
                                                     .perpendicular = false},  //! Label of the region where the radiation is detected by the surface solver
                    (SurfaceRadiationTestParameters){.mpiTestParameter = {.testName = "Perpendicular Plates 1 proc.", .nproc = 1},
                                                     .meshFaces = {10, 10, 10},
                                                     .meshStart = {0, 0, 0},
                                                     .meshEnd = {1, 1, 1},
                                                     .initialization =
                                                         []() {
                                                             return std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>>{
                                                                 std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                                        ablate::mathFunctions::Create("1")),
                                                                 std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                                        ablate::mathFunctions::Create("1000"),
                                                                                                                        nullptr,
                                                                                                                        std::make_shared<ablate::domain::Region>("boundaryCellsLeft")),
                                                                 std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                                        ablate::mathFunctions::Create("1"),
                                                                                                                        nullptr,
                                                                                                                        std::make_shared<ablate::domain::Region>("boundaryCellsTop"))};
                                                         },
                                                     .radiationFactory =
                                                         [](std::shared_ptr<ablate::eos::radiationProperties::RadiationModel> radiationModelIn) {
                                                             auto interiorLabel = std::make_shared<ablate::domain::Region>("interiorCells");
                                                             return std::make_shared<ablate::radiation::SurfaceRadiation>("radiationBase", interiorLabel, 15, radiationModelIn, nullptr);
                                                         },
                                                     .emitLabel = "boundaryCellsLeft",  //! Label of the region from which the radiation is emitted
                                                     .detectLabel = "boundaryCellsTop",
                                                     .perpendicular = true},  //! Label of the region where the radiation is detected by the surface solver
                    (SurfaceRadiationTestParameters){.mpiTestParameter = {.testName = "Parallel Plates 2 proc.", .nproc = 2},
                                                     .meshFaces = {10, 10, 10},
                                                     .meshStart = {0, 0, 0},
                                                     .meshEnd = {1, 1, 1},
                                                     .initialization =
                                                         []() {
                                                             return std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>>{
                                                                 std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                                        ablate::mathFunctions::Create("1")),
                                                                 std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                                        ablate::mathFunctions::Create("1000"),
                                                                                                                        nullptr,
                                                                                                                        std::make_shared<ablate::domain::Region>("boundaryCellsLeft")),
                                                                 std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                                        ablate::mathFunctions::Create("1"),
                                                                                                                        nullptr,
                                                                                                                        std::make_shared<ablate::domain::Region>("boundaryCellsRight"))};
                                                         },
                                                     .radiationFactory =
                                                         [](std::shared_ptr<ablate::eos::radiationProperties::RadiationModel> radiationModelIn) {
                                                             auto interiorLabel = std::make_shared<ablate::domain::Region>("interiorCells");
                                                             return std::make_shared<ablate::radiation::SurfaceRadiation>("radiationBase", interiorLabel, 15, radiationModelIn, nullptr);
                                                         },
                                                     .emitLabel = "boundaryCellsLeft",  //! Label of the region from which the radiation is emitted
                                                     .detectLabel = "boundaryCellsRight",
                                                     .perpendicular = false},  //! Label of the region where the radiation is detected by the surface solver
                    (SurfaceRadiationTestParameters){.mpiTestParameter = {.testName = "Perpendicular Plates 2 proc.", .nproc = 2},
                                                     .meshFaces = {10, 10, 10},
                                                     .meshStart = {0, 0, 0},
                                                     .meshEnd = {1, 1, 1},
                                                     .initialization =
                                                         []() {
                                                             return std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>>{
                                                                 std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                                        ablate::mathFunctions::Create("1")),
                                                                 std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                                        ablate::mathFunctions::Create("1000"),
                                                                                                                        nullptr,
                                                                                                                        std::make_shared<ablate::domain::Region>("boundaryCellsLeft")),
                                                                 std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD,
                                                                                                                        ablate::mathFunctions::Create("1"),
                                                                                                                        nullptr,
                                                                                                                        std::make_shared<ablate::domain::Region>("boundaryCellsTop"))};
                                                         },
                                                     .radiationFactory =
                                                         [](std::shared_ptr<ablate::eos::radiationProperties::RadiationModel> radiationModelIn) {
                                                             auto interiorLabel = std::make_shared<ablate::domain::Region>("interiorCells");
                                                             return std::make_shared<ablate::radiation::SurfaceRadiation>("radiationBase", interiorLabel, 15, radiationModelIn, nullptr);
                                                         },
                                                     .emitLabel = "boundaryCellsLeft",  //! Label of the region from which the radiation is emitted
                                                     .detectLabel = "boundaryCellsTop",
                                                     .perpendicular = true}),  //! Label of the region where the radiation is detected by the surface solver
    [](const testing::TestParamInfo<SurfaceRadiationTestParameters>& info) { return info.param.mpiTestParameter.getTestName(); });