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
#include "finiteVolume/boundaryConditions/ghost.hpp"
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
} BoundarySolverFluxTestParameters;

class BoundarySolverFluxTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<BoundarySolverFluxTestParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

static void FillStencilValues(PetscInt loc, const PetscScalar* stencilValues[], std::vector<PetscScalar>& selectValues) {
    for (std::size_t i = 0; i < selectValues.size(); i++) {
        selectValues[i] = stencilValues[i][loc];
    }
}

TEST_P(BoundarySolverFluxTestFixture, ShouldComputeCorrectGradientsOnBoundary) {
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
            std::make_shared<boundarySolver::BoundarySolver>("testSolver", boundaryCellRegion, boundaryFaceRegion, std::vector<std::shared_ptr<boundarySolver::BoundaryProcess>>{}, nullptr, false);

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

        // Set the boundary boundaryCells values so that they are the correct value on the centroid of the face
        boundarySolver->InsertFieldFunctions(fieldFunctions);
        boundarySolver->InsertFieldFunctions(auxFieldFunctions);

        // Test one boundary point at a time
        PetscReal activeCell[3];
        PetscInt totalDim;
        PetscDSGetTotalDimension(boundarySolver->GetSubDomain().GetDiscreteSystem(), &totalDim) >> checkError;

        // determine the cell size
        PetscReal stencilRadius = .5;

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

                // check for the current active cell
                auto currentActiveCell = (PetscReal*)ctx;

                // Check to see if this is equal
                bool currentCell = true;
                for (PetscInt dir = 0; dir < dim; dir++) {
                    if (PetscAbsReal(boundaryCell->centroid[dir] - currentActiveCell[dir]) > 1E-8) {
                        currentCell = false;
                    }
                }
                if (!currentCell) {
                    return 0;
                }

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
            activeCell,
            {"resultGrad"},
            {"fieldB", "fieldA"},
            {"auxB", "auxA"},
            ablate::boundarySolver::BoundarySolver::BoundarySourceType::Flux);

        // Create a locFVector
        Vec gradVec;
        DMCreateLocalVector(subDomain->GetDM(), &gradVec) >> checkError;

        // Get raw access to the vector
        const PetscScalar* gradArray;
        VecGetArrayRead(gradVec, &gradArray) >> checkError;

        // Get the offset for field
        PetscInt resultGradOffset;
        PetscDSGetFieldOffset(boundarySolver->GetSubDomain().GetDiscreteSystem(), boundarySolver->GetSubDomain().GetField("resultGrad").subId, &resultGradOffset) >> checkError;

        // get the exactGrads
        auto expectedFieldAGradient = ablate::mathFunctions::Create(GetParam().expectedFieldAGradient);
        auto expectedFieldBGradient = ablate::mathFunctions::Create(GetParam().expectedFieldBGradient);
        auto expectedAuxAGradient = ablate::mathFunctions::Create(GetParam().expectedAuxAGradient);
        auto expectedAuxBGradient = ablate::mathFunctions::Create(GetParam().expectedAuxBGradient);

        // Get the list of cells not in the boundary region (i.e. gas phase)
        PetscInt depth;
        DMPlexGetDepth(subDomain->GetDM(), &depth) >> checkError;
        IS allCellIS;
        DMGetStratumIS(subDomain->GetDM(), "depth", depth, &allCellIS) >> checkError;

        // Get the inside cells
        IS insideCellIS;
        IS labelIS;
        DMLabel insideLabel;
        DMGetLabel(subDomain->GetDM(), insideRegion->GetName().c_str(), &insideLabel);
        DMLabelGetStratumIS(insideLabel, insideRegion->GetValue(), &labelIS) >> checkError;
        ISIntersect(allCellIS, labelIS, &insideCellIS) >> checkError;
        ISDestroy(&labelIS) >> checkError;

        // Get the range
        PetscInt insideCellStart, insideCellEnd;
        const PetscInt* insideCells;
        ISGetPointRange(insideCellIS, &insideCellStart, &insideCellEnd, &insideCells);

        // get the cell geometry
        Vec cellGeomVec;
        const PetscScalar* cellGeomArray;
        DM cellGeomDm;
        DMPlexGetDataFVM(subDomain->GetDM(), nullptr, &cellGeomVec, nullptr, nullptr) >> checkError;
        VecGetDM(cellGeomVec, &cellGeomDm) >> checkError;
        VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;

        // March over each cell
        solver::Range boundaryCellRange;
        boundarySolver->GetCellRange(boundaryCellRange);
        for (PetscInt c = boundaryCellRange.start; c < boundaryCellRange.end; ++c) {
            // if there is a cell array, use it, otherwise it is just c
            const PetscInt cell = boundaryCellRange.points ? boundaryCellRange.points[c] : c;

            PetscFVCellGeom* cellGeom;
            DMPlexPointLocalRead(cellGeomDm, cell, cellGeomArray, &cellGeom) >> checkError;

            // Set the current location
            PetscArraycpy(activeCell, cellGeom->centroid, dim);

            // Get the exact location of the face
            const auto& stencils = boundarySolver->GetBoundaryGeometry(cell);
            for (const auto& stencil : stencils) {
                // Reset the grad vec
                VecZeroEntries(gradVec) >> checkError;

                // evaluate
                boundarySolver->ComputeRHSFunction(0.0, globVec, gradVec) >> checkError;

                // Make sure that there is no source terms in this boundary solver region
                for (PetscInt tc = boundaryCellRange.start; tc < boundaryCellRange.end; ++tc) {
                    const PetscInt testCell = boundaryCellRange.points ? boundaryCellRange.points[tc] : tc;

                    const PetscScalar* data;
                    DMPlexPointLocalRead(boundarySolver->GetSubDomain().GetDM(), testCell, gradArray, &data) >> checkError;
                    for (PetscInt i = 0; i < totalDim; i++) {
                        ASSERT_DOUBLE_EQ(0.0, data[i]) << "All at the sources should be zero in the boundarySolverRegion " << testCell;
                    }
                }

                // make sure that the neighbor cell is the one selected
                bool neighborCellFound = false;
                PetscInt neighborCell;
                // March over each face
                PetscInt numberFaces;
                const PetscInt* cellFaces;
                DMPlexGetConeSize(subDomain->GetDM(), cell, &numberFaces) >> checkError;
                DMPlexGetCone(subDomain->GetDM(), cell, &cellFaces) >> checkError;
                for (PetscInt f = 0; f < numberFaces; f++) {
                    PetscInt faceId = cellFaces[f];
                    // Get the connected cells
                    PetscInt numberNeighborCells;
                    const PetscInt* neighborCells;
                    DMPlexGetSupportSize(subDomain->GetDM(), faceId, &numberNeighborCells) >> checkError;
                    DMPlexGetSupport(subDomain->GetDM(), faceId, &neighborCells) >> checkError;
                    if (neighborCells[0] == cell && neighborCells[1] == stencil.stencil.front()) {
                        neighborCellFound = true;
                        neighborCell = neighborCells[1];
                    }
                    if (neighborCells[1] == cell && neighborCells[0] == stencil.stencil.front()) {
                        neighborCellFound = true;
                        neighborCell = neighborCells[0];
                    }
                }

                ASSERT_TRUE(neighborCellFound) << "The first stencil cell for cell " << cell << " is not a itss face neighbor.";

                // The source term * volume on the neighborCell should be the full gradient
                // March over the field sources interiors
                for (PetscInt ic = insideCellStart; ic < insideCellEnd; ++ic) {
                    const PetscInt testCell = insideCells ? insideCells[ic] : ic;

                    const PetscScalar* data;
                    DMPlexPointLocalRead(boundarySolver->GetSubDomain().GetDM(), testCell, gradArray, &data) >> checkError;

                    // If this is not the neighbor cell, make sure that it is zero
                    if (testCell != neighborCell) {
                        for (PetscInt i = 0; i < totalDim; i++) {
                            ASSERT_DOUBLE_EQ(data[i], 0.0) << "Cells not in the neighbor should have no source.";
                        }
                    } else {
                        // compute the volume for the test cell
                        PetscReal volume;
                        DMPlexComputeCellGeometryFVM(boundarySolver->GetSubDomain().GetDM(), testCell, &volume, nullptr, nullptr) >> checkError;

                        // make sure that the grad is equal
                        // Compute the expected values
                        std::vector<PetscReal> exactGradA(3);
                        std::vector<PetscReal> exactGradB(3);
                        std::vector<PetscReal> exactGradAuxA(3);
                        std::vector<PetscReal> exactGradAuxB(3);
                        expectedFieldAGradient->Eval(stencil.geometry.centroid, dim, 0.0, exactGradA);
                        expectedFieldBGradient->Eval(stencil.geometry.centroid, dim, 0.0, exactGradB);
                        expectedAuxAGradient->Eval(stencil.geometry.centroid, dim, 0.0, exactGradAuxA);
                        expectedAuxBGradient->Eval(stencil.geometry.centroid, dim, 0.0, exactGradAuxB);

                        // All the fluxes before the offset should be zero
                        for (PetscInt i = 0; i < resultGradOffset; i++) {
                            ASSERT_DOUBLE_EQ(0.0, data[i]) << "All values not in the 'resultGrad' field should be zero.  Not zero at cell " << cell;
                        }

                        // Now add up the contributions for each cell
                        PetscInt offset = resultGradOffset;

                        // compare the results
                        // March over each field
                        const double absError = 1E-8;
                        for (PetscInt d = 0; d < dim; d++) {
                            ASSERT_NEAR(exactGradA[d], data[offset++] * volume, absError) << "Expected gradient not found for FieldA dir " << d << " in cell " << cell;
                        }
                        for (PetscInt d = 0; d < dim; d++) {
                            ASSERT_NEAR(exactGradB[d], data[offset++] * volume, absError) << "Expected gradient not found for FieldB dir " << d << " in cell " << cell;
                        }
                        for (PetscInt d = 0; d < dim; d++) {
                            ASSERT_NEAR(exactGradAuxA[d], data[offset++] * volume, absError) << "Expected gradient not found for AuxA dir " << d << " in cell " << cell;
                        }
                        for (PetscInt d = 0; d < dim; d++) {
                            ASSERT_NEAR(exactGradAuxB[d], data[offset++] * volume, absError) << "Expected gradient not found for AuxB dir " << d << " in cell " << cell;
                        }
                    }
                }
            }
        }

        boundarySolver->RestoreRange(boundaryCellRange);
        VecRestoreArrayRead(gradVec, &gradArray) >> checkError;

        ISRestorePointRange(insideCellIS, &insideCellStart, &insideCellEnd, &insideCells);

        ISDestroy(&allCellIS);
        ISDestroy(&insideCellIS);
        VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;

        // debug code
        DMViewFromOptions(mesh->GetDM(), nullptr, "-viewTestDM");
        DMViewFromOptions(mesh->GetDM(), nullptr, "-viewTestDMAlso");

        VecDestroy(&gradVec) >> checkError;

        ablate::environment::RunEnvironment::Finalize();
        exit(0);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(BoundarySolver, BoundarySolverFluxTestFixture,
                         testing::Values(
                             (BoundarySolverFluxTestParameters){
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
                             (BoundarySolverFluxTestParameters){
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
                             (BoundarySolverFluxTestParameters){
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
                         [](const testing::TestParamInfo<BoundarySolverFluxTestParameters>& info) { return info.param.mpiTestParameter.getTestName(); });