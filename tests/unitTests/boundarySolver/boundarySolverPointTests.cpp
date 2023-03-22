#include <petsc.h>
#include <cmath>
#include <memory>
#include <vector>
#include "MpiTestFixture.hpp"
#include "boundarySolver/boundarySolver.hpp"
#include "domain/boxMesh.hpp"
#include "domain/modifiers/createLabel.hpp"
#include "domain/modifiers/distributeWithGhostCells.hpp"
#include "domain/modifiers/ghostBoundaryCells.hpp"
#include "domain/modifiers/mergeLabels.hpp"
#include "domain/modifiers/tagLabelBoundary.hpp"
#include "environment/runEnvironment.hpp"
#include "eos/transport/constant.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "mathFunctions/geom/sphere.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscUtilities.hpp"

using namespace ablate;

typedef struct {
    testingResources::MpiTestParameter mpiTestParameter;
    PetscInt dim;
    std::string fieldAFunction;
    std::string fieldBFunction;
    std::string auxAFunction;
    std::string auxBFunction;
    std::string expectedFieldAGradient;
    std::string expectedFieldBGradient;
    std::string expectedAuxAGradient;
    std::string expectedAuxBGradient;
} BoundarySolverPointTestParameters;

class BoundarySolverPointTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<BoundarySolverPointTestParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

static void FillStencilValues(PetscInt loc, const PetscScalar* stencilValues[], std::vector<PetscScalar>& selectValues) {
    for (std::size_t i = 0; i < selectValues.size(); i++) {
        selectValues[i] = stencilValues[i][loc];
    }
}

TEST_P(BoundarySolverPointTestFixture, ShouldComputeCorrectGradientsOnBoundary) {
    StartWithMPI
        // initialize petsc and mpi
        ablate::environment::RunEnvironment::Initialize(argc, argv);
        ablate::utilities::PetscUtilities::Initialize();

        // Define regions for this test
        auto insideRegion = std::make_shared<ablate::domain::Region>("insideRegion");
        auto boundaryFaceRegion = std::make_shared<ablate::domain::Region>("boundaryFaces");
        auto boundaryCellRegion = std::make_shared<ablate::domain::Region>("boundaryCells");
        auto fieldRegion = std::make_shared<ablate::domain::Region>("fieldRegion");

        // define a test field to compute gradients
        std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptors = {
            std::make_shared<ablate::domain::FieldDescription>(
                "fieldA", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::SOL, ablate::domain::FieldType::FVM, fieldRegion),
            std::make_shared<ablate::domain::FieldDescription>(
                "fieldB", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::SOL, ablate::domain::FieldType::FVM, fieldRegion),
            std::make_shared<ablate::domain::FieldDescription>(
                "auxA", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::AUX, ablate::domain::FieldType::FVM, fieldRegion),
            std::make_shared<ablate::domain::FieldDescription>(
                "auxB", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::AUX, ablate::domain::FieldType::FVM, fieldRegion),
            std::make_shared<ablate::domain::FieldDescription>("resultGrad",
                                                               "",
                                                               std::vector<std::string>{"fieldAGrad" + ablate::domain::FieldDescription::DIMENSION,
                                                                                        "fieldBGrad" + ablate::domain::FieldDescription::DIMENSION,
                                                                                        "auxAGrad" + ablate::domain::FieldDescription::DIMENSION,
                                                                                        "auxBGrad" + ablate::domain::FieldDescription::DIMENSION},
                                                               ablate::domain::FieldLocation::SOL,
                                                               ablate::domain::FieldType::FVM,
                                                               fieldRegion)};

        auto dim = GetParam().dim;

        // define the test mesh and setup hthe labels
        auto mesh = std::make_shared<ablate::domain::BoxMesh>(
            "test",
            fieldDescriptors,
            std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{
                std::make_shared<domain::modifiers::DistributeWithGhostCells>(),
                std::make_shared<ablate::domain::modifiers::CreateLabel>(insideRegion, std::make_shared<ablate::mathFunctions::geom::Sphere>(std::vector<double>(dim, .5), .25)),
                std::make_shared<ablate::domain::modifiers::TagLabelBoundary>(insideRegion, boundaryFaceRegion, boundaryCellRegion),
                std::make_shared<ablate::domain::modifiers::MergeLabels>(fieldRegion, std::vector<std::shared_ptr<domain::Region>>{insideRegion, boundaryCellRegion}),
                std::make_shared<domain::modifiers::GhostBoundaryCells>()},
            std::vector<int>(dim, 5),
            std::vector<double>(dim, 0.0),
            std::vector<double>(dim, 1.0),
            std::vector<std::string>(dim, "NONE") /*boundary*/,
            true /*simplex*/);

        // create a boundarySolver
        auto boundarySolver =
            std::make_shared<boundarySolver::BoundarySolver>("testSolver", boundaryCellRegion, boundaryFaceRegion, std::vector<std::shared_ptr<boundarySolver::BoundaryProcess>>{}, nullptr, true);

        // Init the subDomain
        mesh->InitializeSubDomains({boundarySolver}, {});

        // Get the global vectors
        auto globVec = mesh->GetSolutionVector();

        // Initialize each of the fields
        auto subDomain = mesh->GetSubDomain(boundaryCellRegion);
        auto fieldFunctions = {
            std::make_shared<mathFunctions::FieldFunction>("fieldA", ablate::mathFunctions::Create(GetParam().fieldAFunction)),
            std::make_shared<mathFunctions::FieldFunction>("fieldB", ablate::mathFunctions::Create(GetParam().fieldBFunction)),
        };
        mesh->ProjectFieldFunctions(fieldFunctions, globVec);

        auto auxVec = subDomain->GetAuxVector();
        auto auxFieldFunctions = {
            std::make_shared<mathFunctions::FieldFunction>("auxA", ablate::mathFunctions::Create(GetParam().auxAFunction)),
            std::make_shared<mathFunctions::FieldFunction>("auxB", ablate::mathFunctions::Create(GetParam().auxBFunction)),
        };
        subDomain->ProjectFieldFunctionsToLocalVector(auxFieldFunctions, auxVec);

        // Set the boundary cells values so that they are the correct value on the centroid of the face
        boundarySolver->InsertFieldFunctions(fieldFunctions);
        boundarySolver->InsertFieldFunctions(auxFieldFunctions);

        // for each
        boundarySolver->RegisterFunction(
            [](PetscInt dim,
               const boundarySolver::BoundarySolver::BoundaryFVFaceGeom* fg,
               const PetscFVCellGeom* boundaryCell,
               const PetscInt uOff[],
               const PetscScalar* boundaryValues,
               const PetscScalar* stencilValues[],
               const PetscInt aOff[],
               const PetscScalar* auxValues,
               const PetscScalar* stencilAuxValues[],
               PetscInt stencilSize,
               const PetscInt stencil[],
               const PetscScalar stencilWeights[],
               const PetscInt sOff[],
               PetscScalar source[],
               void* ctx) {
                const PetscInt fieldA = 1;
                const PetscInt fieldB = 0;
                const PetscInt sourceField = 0;
                const PetscInt auxA = 1;
                const PetscInt auxB = 0;

                // Create a scratch space
                std::vector<PetscScalar> pointValues(stencilSize, 0.0);
                PetscInt sourceOffset = 0;

                // Compute each field
                FillStencilValues(uOff[fieldA], stencilValues, pointValues);
                boundarySolver::BoundarySolver::ComputeGradient(dim, boundaryValues[uOff[fieldA]], stencilSize, &pointValues[0], stencilWeights, source + sOff[sourceField] + (sourceOffset * dim));

                // Check this gradient with the dPhiDn Calc
                PetscScalar dPhiDNorm;
                boundarySolver::BoundarySolver::ComputeGradientAlongNormal(dim, fg, boundaryValues[uOff[fieldA]], stencilSize, &pointValues[0], stencilWeights, dPhiDNorm);
                if (PetscAbs(dPhiDNorm - (utilities::MathUtilities::DotVector(dim, source + sOff[sourceField] + (sourceOffset * dim), fg->normal))) > 1E-8) {
                    throw std::runtime_error("The ComputeGradientAlongNormal function computed a wrong gradient for boundaryValues[uOff[fieldA]]");
                }

                FillStencilValues(uOff[fieldB], stencilValues, pointValues);
                sourceOffset++;
                boundarySolver::BoundarySolver::ComputeGradient(dim, boundaryValues[uOff[fieldB]], stencilSize, &pointValues[0], stencilWeights, source + sOff[sourceField] + (sourceOffset * dim));
                boundarySolver::BoundarySolver::ComputeGradientAlongNormal(dim, fg, boundaryValues[uOff[fieldB]], stencilSize, &pointValues[0], stencilWeights, dPhiDNorm);
                if (PetscAbs(dPhiDNorm - (utilities::MathUtilities::DotVector(dim, source + sOff[sourceField] + (sourceOffset * dim), fg->normal))) > 1E-8) {
                    throw std::runtime_error("The ComputeGradientAlongNormal function computed a wrong gradient for boundaryValues[uOff[fieldB]]");
                }

                FillStencilValues(uOff[auxA], stencilAuxValues, pointValues);
                sourceOffset++;
                boundarySolver::BoundarySolver::ComputeGradient(dim, auxValues[aOff[auxA]], stencilSize, &pointValues[0], stencilWeights, source + sOff[sourceField] + (sourceOffset * dim));
                boundarySolver::BoundarySolver::ComputeGradientAlongNormal(dim, fg, auxValues[aOff[auxA]], stencilSize, &pointValues[0], stencilWeights, dPhiDNorm);
                if (PetscAbs(dPhiDNorm - (utilities::MathUtilities::DotVector(dim, source + sOff[sourceField] + (sourceOffset * dim), fg->normal))) > 1E-8) {
                    throw std::runtime_error("The ComputeGradientAlongNormal function computed a wrong gradient for auxValues[aOff[auxA]]");
                }

                FillStencilValues(uOff[auxB], stencilAuxValues, pointValues);
                sourceOffset++;
                boundarySolver::BoundarySolver::ComputeGradient(dim, auxValues[aOff[auxB]], stencilSize, &pointValues[0], stencilWeights, source + sOff[sourceField] + (sourceOffset * dim));
                boundarySolver::BoundarySolver::ComputeGradientAlongNormal(dim, fg, auxValues[aOff[auxB]], stencilSize, &pointValues[0], stencilWeights, dPhiDNorm);
                if (PetscAbs(dPhiDNorm - (utilities::MathUtilities::DotVector(dim, source + sOff[sourceField] + (sourceOffset * dim), fg->normal))) > 1E-8) {
                    throw std::runtime_error("The ComputeGradientAlongNormal function computed a wrong gradient for auxValues[aOff[auxB]]");
                }

                // The normal should point away from the center of the domain
                PetscScalar center[3] = {.5, .5, .5};
                PetscScalar outwardVector[3];
                for (PetscInt d = 0; d < dim; d++) {
                    outwardVector[d] = fg->centroid[d] - center[d];
                }
                if (utilities::MathUtilities::DotVector(dim, outwardVector, fg->normal) <= 0) {
                    throw std::runtime_error("The Normal should face out from the inside region.");
                }

                return 0;
            },
            nullptr,
            {"resultGrad"},
            {"fieldB", "fieldA"},
            {"auxB", "auxA"});

        // Create a locFVector
        Vec gradVec;
        DMCreateLocalVector(subDomain->GetDM(), &gradVec) >> utilities::PetscUtilities::checkError;

        // evaluate
        boundarySolver->ComputeRHSFunction(0.0, globVec, gradVec);

        // Get raw access to the vector
        const PetscScalar* gradArray;
        VecGetArrayRead(gradVec, &gradArray) >> utilities::PetscUtilities::checkError;

        // Get the offset for field
        PetscInt resultGradOffset;
        PetscDSGetFieldOffset(boundarySolver->GetSubDomain().GetDiscreteSystem(), boundarySolver->GetSubDomain().GetField("resultGrad").subId, &resultGradOffset) >>
            utilities::PetscUtilities::checkError;

        // get the exactGrads
        auto expectedFieldAGradient = ablate::mathFunctions::Create(GetParam().expectedFieldAGradient);
        auto expectedFieldBGradient = ablate::mathFunctions::Create(GetParam().expectedFieldBGradient);
        auto expectedAuxAGradient = ablate::mathFunctions::Create(GetParam().expectedAuxAGradient);
        auto expectedAuxBGradient = ablate::mathFunctions::Create(GetParam().expectedAuxBGradient);

        // March over each cell
        ablate::domain::Range cellRange;
        boundarySolver->GetCellRange(cellRange);
        PetscInt cOffset = 0;  // Keep track of the cell offset
        for (PetscInt c = cellRange.start; c < cellRange.end; ++c, cOffset++) {
            // if there is a cell array, use it, otherwise it is just c
            const PetscInt cell = cellRange.points ? cellRange.points[c] : c;

            // Get the raw data at this point, this check assumes the order the fields
            const PetscScalar* data;
            DMPlexPointLocalRead(boundarySolver->GetSubDomain().GetDM(), cell, gradArray, &data) >> utilities::PetscUtilities::checkError;

            // All the fluxes before the offset should be zero
            for (PetscInt i = 0; i < resultGradOffset; i++) {
                ASSERT_DOUBLE_EQ(0.0, data[i]) << "All values not in the 'resultGrad' field should be zero.  Not zero at cell " << cell;
            }

            // Get the exact location of the face
            const auto& faces = boundarySolver->GetBoundaryGeometry(cell);
            const auto& face = faces.front();

            // March over each source and compare against the known solution assuming field order
            PetscInt offset = resultGradOffset;
            std::vector<PetscScalar> exactGrad(3);

            // March over each field
            const double absError = 1E-8;
            expectedFieldAGradient->Eval(face.geometry.centroid, dim, 0.0, exactGrad);
            for (PetscInt d = 0; d < dim; d++) {
                ASSERT_NEAR(exactGrad[d], data[offset++], absError) << "Expected gradient not found for FieldA dir " << d << " in cell " << cell;
            }
            expectedFieldBGradient->Eval(face.geometry.centroid, dim, 0.0, exactGrad);
            for (PetscInt d = 0; d < dim; d++) {
                ASSERT_NEAR(exactGrad[d], data[offset++], absError) << "Expected gradient not found for FieldB dir " << d << " in cell " << cell;
            }
            expectedAuxAGradient->Eval(face.geometry.centroid, dim, 0.0, exactGrad);
            for (PetscInt d = 0; d < dim; d++) {
                ASSERT_NEAR(exactGrad[d], data[offset++], absError) << "Expected gradient not found for AuxA dir " << d << " in cell " << cell;
            }
            expectedAuxBGradient->Eval(face.geometry.centroid, dim, 0.0, exactGrad);
            for (PetscInt d = 0; d < dim; d++) {
                ASSERT_NEAR(exactGrad[d], data[offset++], absError) << "Expected gradient not found for AuxB dir " << d << " in cell " << cell;
            }
        }

        boundarySolver->RestoreRange(cellRange);
        VecRestoreArrayRead(gradVec, &gradArray) >> utilities::PetscUtilities::checkError;

        // debug code
        DMViewFromOptions(mesh->GetDM(), nullptr, "-viewTestDM");
        DMViewFromOptions(mesh->GetDM(), nullptr, "-viewTestDMAlso");

        VecDestroy(&gradVec) >> utilities::PetscUtilities::checkError;

        ablate::environment::RunEnvironment::Finalize();
        exit(0);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(BoundarySolver, BoundarySolverPointTestFixture,
                         testing::Values(
                             (BoundarySolverPointTestParameters){
                                 .mpiTestParameter = {.testName = "1D BoundarySolver", .nproc = 1, .arguments = ""},
                                 .dim = 1,
                                 .fieldAFunction = "x + x*y+ y + z",
                                 .fieldBFunction = "10*x + 3*y + z*x +2*z",
                                 .auxAFunction = "-x - y -z",
                                 .auxBFunction = "-x*y*z",
                                 .expectedFieldAGradient = "1 + y, x + 1, 1",
                                 .expectedFieldBGradient = "10+z, 3, x + 2",
                                 .expectedAuxAGradient = "-1, -1, -1",
                                 .expectedAuxBGradient = "-y*z, -x*z, -x*y",

                             },
                             (BoundarySolverPointTestParameters){
                                 .mpiTestParameter = {.testName = "2D BoundarySolver", .nproc = 1, .arguments = ""},
                                 .dim = 2,
                                 .fieldAFunction = "x + y + z",
                                 .fieldBFunction = "10*x + 3*y +2*z",
                                 .auxAFunction = "-x - y -z",
                                 .auxBFunction = "-x-x",
                                 .expectedFieldAGradient = "1,  1, 1",
                                 .expectedFieldBGradient = "10, 3,  2",
                                 .expectedAuxAGradient = "-1, -1, -1",
                                 .expectedAuxBGradient = "-2,0, 0",

                             },
                             (BoundarySolverPointTestParameters){
                                 .mpiTestParameter = {.testName = "3D BoundarySolver", .nproc = 1, .arguments = ""},
                                 .dim = 3,
                                 .fieldAFunction = "x + y + z",
                                 .fieldBFunction = "10*x + 3*y +2*z",
                                 .auxAFunction = "-x - y -z",
                                 .auxBFunction = "-x-x",
                                 .expectedFieldAGradient = "1,  1, 1",
                                 .expectedFieldBGradient = "10, 3,  2",
                                 .expectedAuxAGradient = "-1, -1, -1",
                                 .expectedAuxBGradient = "-2,0, 0",

                             }),
                         [](const testing::TestParamInfo<BoundarySolverPointTestParameters>& info) { return info.param.mpiTestParameter.getTestName(); });
