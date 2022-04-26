#include <petsc.h>
#include <cmath>
#include <memory>
#include <vector>
#include "MpiTestFixture.hpp"
#include "PetscTestErrorChecker.hpp"
#include "boundarySolver/boundarySolver.hpp"
#include "domain/boxMesh.hpp"
#include "domain/modifiers/createLabel.hpp"
#include "domain/modifiers/distributeWithGhostCells.hpp"
#include "domain/modifiers/ghostBoundaryCells.hpp"
#include "domain/modifiers/mergeLabels.hpp"
#include "domain/modifiers/tagLabelBoundary.hpp"
#include "eos/transport/constant.hpp"
#include "finiteVolume/boundaryConditions/ghost.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "mathFunctions/geom/sphere.hpp"
#include "utilities/mathUtilities.hpp"

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
} FaceInterpolantTestParameters;

class FaceInterpolantTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<FaceInterpolantTestParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

TEST_P(FaceInterpolantTestFixture, ShouldComputeCorrectGradientsOnBoundary) {
    StartWithMPI
        // initialize petsc and mpi
        PetscInitialize(argc, argv, nullptr, "HELP") >> testErrorChecker;

        // get the exactGrads
        auto expectedFieldA = ablate::mathFunctions::Create(GetParam().fieldAFunction);
        auto expectedFieldB = ablate::mathFunctions::Create(GetParam().fieldBFunction);
        auto expectedAuxA = ablate::mathFunctions::Create(GetParam().auxAFunction);
        auto expectedAuxB = ablate::mathFunctions::Create(GetParam().auxBFunction);
        auto expectedFieldAGradient = ablate::mathFunctions::Create(GetParam().expectedFieldAGradient);
        auto expectedFieldBGradient = ablate::mathFunctions::Create(GetParam().expectedFieldBGradient);
        auto expectedAuxAGradient = ablate::mathFunctions::Create(GetParam().expectedAuxAGradient);
        auto expectedAuxBGradient = ablate::mathFunctions::Create(GetParam().expectedAuxBGradient);

        // define a test field to compute gradients
        std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptors = {
            std::make_shared<ablate::domain::FieldDescription>("fieldA", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::SOL, ablate::domain::FieldType::FVM),
            std::make_shared<ablate::domain::FieldDescription>("fieldB", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::SOL, ablate::domain::FieldType::FVM),
            std::make_shared<ablate::domain::FieldDescription>("auxA", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::AUX, ablate::domain::FieldType::FVM),
            std::make_shared<ablate::domain::FieldDescription>("auxB", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::AUX, ablate::domain::FieldType::FVM)};

        auto dim = GetParam().dim;

        // define the test mesh and set up the labels
        auto mesh = std::make_shared<ablate::domain::BoxMesh>("test",
                                                              fieldDescriptors,
                                                              std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{},
                                                              std::vector<int>(dim, 5),
                                                              std::vector<double>(dim, 0.0),
                                                              std::vector<double>(dim, 1.0),
                                                              std::vector<std::string>(dim, "NONE") /*boundary*/,
                                                              false /*simplex*/);
        DMCreateLabel(mesh->GetDM(), "ghost");

        // debug code
        DMViewFromOptions(mesh->GetDM(), nullptr, "-viewTestDM");
        DMViewFromOptions(mesh->GetDM(), nullptr, "-viewTestDMAlso");

        // create a boundarySolver
        auto fvSolver = std::make_shared<finiteVolume::FiniteVolumeSolver>("testSolver",
                                                                           domain::Region::ENTIREDOMAIN,
                                                                           nullptr,
                                                                           std::vector<std::shared_ptr<finiteVolume::processes::Process>>{},
                                                                           std::vector<std::shared_ptr<finiteVolume::boundaryConditions::BoundaryCondition>>{});

        // Init the subDomain
        mesh->InitializeSubDomains({fvSolver}, {});

        // Get the mesh cell information
        Vec cellGeomVec, faceGeomVec;
        DMPlexComputeGeometryFVM(mesh->GetSubDomain(domain::Region::ENTIREDOMAIN)->GetDM(), &cellGeomVec, &faceGeomVec) >> testErrorChecker;

        // create a test faceInterpolant
        ablate::finiteVolume::FaceInterpolant faceInterpolant(mesh->GetSubDomain(domain::Region::ENTIREDOMAIN), faceGeomVec, cellGeomVec);

        // Get the global vectors
        auto globVec = mesh->GetSolutionVector();

        // Initialize each of the fields
        auto subDomain = mesh->GetSubDomain(domain::Region::ENTIREDOMAIN);
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

        // interpolate to the faces
        Vec faceSolutionVec, faceAuxVec, faceSolutionGradVec, faceAuxGradVec;
        faceInterpolant.GetInterpolatedFaceVectors(subDomain->GetSolutionVector(), auxVec, faceSolutionVec, faceAuxVec, faceSolutionGradVec, faceAuxGradVec);

        // run over each face to see if it computed the gradient correctly
        solver::Range faceRange;
        fvSolver->GetFaceRange(faceRange);

        // extract the arrays for each of the vec
        DM faceSolutionDm, faceAuxDm;
        const PetscScalar* faceSolutionArray;
        const PetscScalar* faceAuxArray;
        VecGetArrayRead(faceSolutionVec, &faceSolutionArray);
        VecGetArrayRead(faceAuxVec, &faceAuxArray);
        VecGetDM(faceSolutionVec, &faceSolutionDm);
        VecGetDM(faceAuxVec, &faceAuxDm);

        DM faceSolutionGradDm, faceAuxGradDm;
        const PetscScalar* faceSolutionGradArray;
        const PetscScalar* faceAuxGradArray;
        VecGetArrayRead(faceSolutionGradVec, &faceSolutionGradArray);
        VecGetArrayRead(faceAuxGradVec, &faceAuxGradArray);
        VecGetDM(faceSolutionGradVec, &faceSolutionGradDm);
        VecGetDM(faceAuxGradVec, &faceAuxGradDm);

        // Get the geometry for the mesh
        DM faceDM;
        VecGetDM(faceGeomVec, &faceDM) >> checkError;
        const PetscScalar* faceGeomArray;
        VecGetArrayRead(faceGeomVec, &faceGeomArray) >> checkError;

        // march over each face
        for (PetscInt f = faceRange.start; f < faceRange.end; f++) {
            PetscInt face = faceRange.points ? faceRange.points[f] : f;

            // Get the face geom location
            PetscFVFaceGeom* fg;
            DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg) >> testErrorChecker;

            PetscInt supportSize;
            DMPlexGetSupportSize(subDomain->GetDM(), face, &supportSize);
            if (supportSize != 2) {
                continue;
            }

            // March over each source and compare against the known solution assuming field order
            std::vector<PetscScalar> exactGrad(3);

            // March over each field
            const double absError = 1E-8;

            {  // fieldA
                const PetscScalar* value;
                DMPlexPointLocalRead(faceSolutionDm, face, faceSolutionArray, &value);
                ASSERT_NEAR(expectedFieldA->Eval(fg->centroid, dim, 0.0), value[0], absError) << "Expected value not found for FieldA at face " << face;
                const PetscScalar* gradValue;
                DMPlexPointLocalRead(faceSolutionGradDm, face, faceSolutionGradArray, &gradValue);
                expectedFieldAGradient->Eval(fg->centroid, dim, 0.0, exactGrad);
                for (PetscInt d = 0; d < dim; d++) {
                    ASSERT_NEAR(exactGrad[d], gradValue[0 + d], absError) << "Expected gradient not found for FieldA dir " << d << " at face " << face;
                }
            }
            {  // fieldB
                const PetscScalar* value;
                DMPlexPointLocalRead(faceSolutionDm, face, faceSolutionArray, &value);
                ASSERT_NEAR(expectedFieldB->Eval(fg->centroid, dim, 0.0), value[1], absError) << "Expected value not found for FieldB at face " << face;
                const PetscScalar* gradValue;
                DMPlexPointLocalRead(faceSolutionGradDm, face, faceSolutionGradArray, &gradValue);
                expectedFieldBGradient->Eval(fg->centroid, dim, 0.0, exactGrad);
                for (PetscInt d = 0; d < dim; d++) {
                    ASSERT_NEAR(exactGrad[d], gradValue[dim + d], absError) << "Expected gradient not found for FieldB dir " << d << " at face " << face;
                }
            }
            {  // auxA
                const PetscScalar* value;
                DMPlexPointLocalRead(faceAuxDm, face, faceAuxArray, &value);
                ASSERT_NEAR(expectedAuxA->Eval(fg->centroid, dim, 0.0), value[0], absError) << "Expected value not found for AuxA at face " << face;
                const PetscScalar* gradValue;
                DMPlexPointLocalRead(faceAuxGradDm, face, faceAuxGradArray, &gradValue);
                expectedAuxAGradient->Eval(fg->centroid, dim, 0.0, exactGrad);
                for (PetscInt d = 0; d < dim; d++) {
                    ASSERT_NEAR(exactGrad[d], gradValue[0 + d], absError) << "Expected gradient not found for AuxA dir " << d << " at face " << face;
                }
            }
            {  // auxB
                const PetscScalar* value;
                DMPlexPointLocalRead(faceAuxDm, face, faceAuxArray, &value);
                ASSERT_NEAR(expectedAuxB->Eval(fg->centroid, dim, 0.0), value[1], absError) << "Expected value not found for AuxB at face " << face;
                const PetscScalar* gradValue;
                DMPlexPointLocalRead(faceAuxGradDm, face, faceAuxGradArray, &gradValue);
                expectedAuxBGradient->Eval(fg->centroid, dim, 0.0, exactGrad);
                for (PetscInt d = 0; d < dim; d++) {
                    ASSERT_NEAR(exactGrad[d], gradValue[dim + d], absError) << "Expected gradient not found for AuxB dir " << d << " at face " << face;
                }
            }
        }

        VecRestoreArrayRead(faceSolutionGradVec, &faceSolutionGradArray);
        VecRestoreArrayRead(faceAuxGradVec, &faceAuxGradArray);
        VecRestoreArrayRead(faceSolutionVec, &faceSolutionArray);
        VecRestoreArrayRead(faceAuxVec, &faceAuxArray);
        VecRestoreArrayRead(faceGeomVec, &faceGeomArray);
        fvSolver->RestoreRange(faceRange);

        exit(PetscFinalize());
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(FaceInterpolant, FaceInterpolantTestFixture,
                         testing::Values(
                             (FaceInterpolantTestParameters){
                                 .mpiTestParameter = {.testName = "1D BoundarySolver", .nproc = 1, .arguments = ""},
                                 .dim = 1,
                                 .fieldAFunction = "x + y + z",
                                 .fieldBFunction = "10*x + 3*y + z*x +2*z",
                                 .auxAFunction = "-x - y -z",
                                 .auxBFunction = "-x*y*z",
                                 .expectedFieldAGradient = "1, 1, 1",
                                 .expectedFieldBGradient = "10+z, 3, x + 2",
                                 .expectedAuxAGradient = "-1, -1, -1",
                                 .expectedAuxBGradient = "-y*z, -x*z, -x*y",

                             },
                             (FaceInterpolantTestParameters){
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
                             (FaceInterpolantTestParameters){
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
                         [](const testing::TestParamInfo<FaceInterpolantTestParameters>& info) { return info.param.mpiTestParameter.getTestName(); });