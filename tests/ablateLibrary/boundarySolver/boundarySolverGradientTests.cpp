#include <petsc.h>
#include <boundarySolver/boundarySolver.hpp>
#include <cmath>
#include <domain/modifiers/createLabel.hpp>
#include <domain/modifiers/mergeLabels.hpp>
#include <domain/modifiers/tagLabelBoundary.hpp>
#include <mathFunctions/geom/sphere.hpp>
#include <memory>
#include <vector>
#include "MpiTestFixture.hpp"
#include "PetscTestErrorChecker.hpp"
#include "domain/boxMesh.hpp"
#include "domain/modifiers/distributeWithGhostCells.hpp"
#include "domain/modifiers/ghostBoundaryCells.hpp"
#include "eos/transport/constant.hpp"
#include "finiteVolume/boundaryConditions/ghost.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"

using namespace ablate;

typedef struct {
    testingResources::MpiTestParameter mpiTestParameter;
    PetscInt dim;
} BoundarySolverGradientTestParameters;

class BoundarySolverGradientTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<BoundarySolverGradientTestParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

static void FillStencilValues(PetscInt loc, const PetscScalar* stencilValues[], std::vector<PetscScalar>& selectValues){
    for(std::size_t i =0; i < selectValues.size(); i++){
        selectValues[i] = stencilValues[i][loc];
    }
}

TEST_P(BoundarySolverGradientTestFixture, ShouldComputeCorrectGradientsOnBoundary) {
    StartWithMPI
        // initialize petsc and mpi
        PetscInitialize(argc, argv, NULL, "HELP") >> testErrorChecker;

        // Define regions for this test
        auto insideRegion = std::make_shared<ablate::domain::Region>("insideRegion");
        auto boundaryFaceRegion = std::make_shared<ablate::domain::Region>("boundaryFaces");
        auto boundaryCellRegion = std::make_shared<ablate::domain::Region>("boundaryCells");
        auto fieldRegion = std::make_shared<ablate::domain::Region>("fieldRegion");

        // define a test field to compute gradients
        std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptors = {
            std::make_shared<ablate::domain::FieldDescription>("fieldA", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::SOL, ablate::domain::FieldType::FVM, fieldRegion),
            std::make_shared<ablate::domain::FieldDescription>(
                "fieldB", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::SOL, ablate::domain::FieldType::FVM, fieldRegion),
            std::make_shared<ablate::domain::FieldDescription>(
                "auxA", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::AUX, ablate::domain::FieldType::FVM, fieldRegion),
            std::make_shared<ablate::domain::FieldDescription>("auxB", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::AUX, ablate::domain::FieldType::FVM, fieldRegion),
            std::make_shared<ablate::domain::FieldDescription>("resultGrad", "", std::vector<std::string>{"fieldAGrad" + ablate::domain::FieldDescription::DIMENSION, "fieldBGrad" + ablate::domain::FieldDescription::DIMENSION, "auxAGrad" + ablate::domain::FieldDescription::DIMENSION, "auxBGrad" + ablate::domain::FieldDescription::DIMENSION}, ablate::domain::FieldLocation::SOL, ablate::domain::FieldType::FVM, fieldRegion)
        };

        auto dim = GetParam().dim;

        // define the test mesh and setup hthe labels
        auto mesh = std::make_shared<ablate::domain::BoxMesh>(
            "test",
            fieldDescriptors,
            std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{

                std::make_shared<domain::modifiers::DistributeWithGhostCells>(),
                std::make_shared<domain::modifiers::GhostBoundaryCells>(),
                std::make_shared<ablate::domain::modifiers::CreateLabel>(insideRegion, std::make_shared<ablate::mathFunctions::geom::Sphere>(std::vector<double>(dim, .5), .25)),
                std::make_shared<ablate::domain::modifiers::TagLabelBoundary>(insideRegion, boundaryFaceRegion, boundaryCellRegion),
                std::make_shared<ablate::domain::modifiers::MergeLabels>(fieldRegion, std::vector<std::shared_ptr<domain::Region>>{insideRegion, boundaryCellRegion})},
            std::vector<int>(dim, 5),
            std::vector<double>(dim, 0.0),
            std::vector<double>(dim, 1.0),
            std::vector<std::string>(dim, "NONE") /*boundary*/,
            true /*simplex*/);

        // create a boundarySolver
        auto boundarySolver =
            std::make_shared<boundarySolver::BoundarySolver>("testSolver", boundaryCellRegion, boundaryFaceRegion, std::vector<std::shared_ptr<boundarySolver::BoundaryProcess>>{}, nullptr);

        // Init the subDomain
        mesh->InitializeSubDomains({boundarySolver});

        // Get the global vectors
        auto globVec = mesh->GetSolutionVector();

        // Initialize each of the fields
        auto subDomain = mesh->GetSubDomain(boundaryCellRegion);
        auto fieldFunctions = {
            std::make_shared<mathFunctions::FieldFunction>("fieldA", ablate::mathFunctions::Create("x + y + z")),
            std::make_shared<mathFunctions::FieldFunction>("fieldB", ablate::mathFunctions::Create("x^2 + y^2 + z^2")),
        };
        subDomain->ProjectFieldFunctions(fieldFunctions, globVec);

        auto auxVec = subDomain->GetAuxVector();
        auto auxFieldFunctions = {
            std::make_shared<mathFunctions::FieldFunction>("auxA", ablate::mathFunctions::Create("10*x + 10*y + 10*z")),
            std::make_shared<mathFunctions::FieldFunction>("auxB", ablate::mathFunctions::Create("10*x^2 + 10*y^2 + 10*z^2")),
        };
        subDomain->ProjectFieldFunctions(auxFieldFunctions, auxVec);

        // Set the boundary cells values so that they are the correct value on the centroid of the face
        boundarySolver->InsertFieldFunctions(fieldFunctions);
        boundarySolver->InsertFieldFunctions(auxFieldFunctions);

        // for each
        boundarySolver->RegisterFunction([](PetscInt dim, const boundarySolver::BoundarySolver::BoundaryFVFaceGeom* fg, const PetscFVCellGeom* boundaryCell,
                                                           const PetscInt uOff[], const PetscScalar* boundaryValues, const PetscScalar* stencilValues[],
                                                           const PetscInt aOff[], const PetscScalar* auxValues, const PetscScalar* stencilAuxValues[],
                                                           PetscInt stencilSize, const PetscInt stencil[], const PetscScalar stencilWeights[], const PetscInt sOff[], PetscScalar source[], void* ctx){

            const PetscInt fieldA = 1;
            const PetscInt fieldB = 0;
            const PetscInt auxA = 1;
            const PetscInt auxB= 0;

            // Create a scratch space
            std::vector<PetscScalar> pointValues(stencilSize, 0.0);
            PetscInt sourceOffset = 0;

            // Compute each field
            FillStencilValues(uOff[fieldA], stencilValues, pointValues);
            boundarySolver::BoundarySolver::ComputeGradient(dim, boundaryValues[uOff[fieldA]], stencilSize, &pointValues[0], stencilWeights, source + (sourceOffset++ *dim));

            FillStencilValues(uOff[fieldB], stencilValues, pointValues);
            boundarySolver::BoundarySolver::ComputeGradient(dim, boundaryValues[uOff[fieldB]], stencilSize, &pointValues[0], stencilWeights, source + (sourceOffset++ *dim));

            FillStencilValues(uOff[auxA], stencilAuxValues, pointValues);
            boundarySolver::BoundarySolver::ComputeGradient(dim, auxValues[aOff[auxA]], stencilSize, &pointValues[0], stencilWeights, source + (sourceOffset++ *dim));

            FillStencilValues(uOff[auxB], stencilAuxValues, pointValues);
            boundarySolver::BoundarySolver::ComputeGradient(dim, auxValues[aOff[auxB]], stencilSize, &pointValues[0], stencilWeights, source + (sourceOffset++ *dim));

            return 0;
        }, nullptr, {"resultGrad"}, { "fieldB", "fieldA"}, {"auxB", "auxA"});

        // Create a locFVector
        Vec gradVec;
        DMCreateLocalVector(subDomain->GetDM(), &gradVec) >> checkError;

        // evaluate
        boundarySolver->ComputeRHSFunction(0.0, globVec, gradVec);

        // extract the gradients from each


        // debug code
        DMViewFromOptions(mesh->GetDM(), nullptr, "-testDM");
        VecView(gradVec, PETSC_VIEWER_STDOUT_WORLD);

        VecDestroy(&gradVec) >> checkError;

        exit(PetscFinalize());
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(BoundarySolver, BoundarySolverGradientTestFixture,
                        testing::Values((BoundarySolverGradientTestParameters){.mpiTestParameter = {.testName = "1D BoundarySolver", .nproc = 1, .arguments = ""},
                                                                                .dim = 1}),
                        [](const testing::TestParamInfo<BoundarySolverGradientTestParameters>& info) { return info.param.mpiTestParameter.getTestName(); });