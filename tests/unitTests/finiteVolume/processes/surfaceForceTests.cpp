#include <petsc.h>
#include <mathFunctions/functionFactory.hpp>
#include <memory>
#include <vector>
#include "MpiTestFixture.hpp"
#include "domain/boxMesh.hpp"
#include "domain/modifiers/ghostBoundaryCells.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "finiteVolume/processes/surfaceForce.hpp"
#include "finiteVolume/processes/twoPhaseEulerAdvection.hpp"
#include "gtest/gtest.h"

struct SurfaceForceTestParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    PetscInt dim;
    PetscInt cellNumber;
    std::vector<int> meshFaces;
    std::vector<double> meshStart;
    std::vector<double> meshEnd;
    std::vector<PetscReal> inputEulerValues;
    std::shared_ptr<ablate::mathFunctions::MathFunction> inputVFfield;
    std::vector<PetscReal> expectedEulerSource;
    PetscReal errorTolerance = 1E-3;
    PetscReal sigma = 0.07;
};

class SurfaceForceTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<SurfaceForceTestParameters> {};

TEST_P(SurfaceForceTestFixture, ShouldComputeCorrectSurfaceForce) {
    ablate::utilities::PetscUtilities::Initialize();
    const auto &params = GetParam();

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
    auto domain = std::make_shared<ablate::domain::BoxMesh>("test",

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

    ablate::finiteVolume::processes::SurfaceForce::VertexStencil stencilData;
    std::vector<ablate::finiteVolume::processes::SurfaceForce::VertexStencil> vertexStencils;

    DM dm;
    dm = domain->GetDM();

    Vec cellGeomVec;
    DM dmCell;
    const PetscScalar *cellGeomArray;
    DMPlexGetGeometryFVM(dm, nullptr, &cellGeomVec, nullptr);
    VecGetDM(cellGeomVec, &dmCell);
    VecGetArrayRead(cellGeomVec, &cellGeomArray);

    // extract the local x array
    Vec localCoordsVector;
    PetscSection coordsSection;
    PetscScalar *coordsArray;
    DMGetCoordinateSection(dm, &coordsSection);
    DMGetCoordinatesLocal(dm, &localCoordsVector);
    VecGetArray(localCoordsVector, &coordsArray);

    PetscInt vStart, vEnd, cStart, cEnd, cl;
    // extract vortices of domain
    DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);

    // march over vortices
    for (PetscInt v = vStart; v < vEnd; v++) {
        // store the vortex point
        stencilData.vertexId = v;
        stencilData.stencilCoord = {0, 0, 0};
        PetscInt off;
        PetscReal xyz[3];
        // extract x, y, z values
        PetscSectionGetOffset(coordsSection, v, &off);
        for (PetscInt d = 0; d < dim; ++d) {
            xyz[d] = coordsArray[off + d];
            // store coordinates of the vortex
            stencilData.stencilCoord[d] = xyz[d];
        }

        stencilData.gradientWeights = {0, 0, 0};
        stencilData.stencilSize = 0;
        // PetscInt nCell =0;
        //  extract faces of the vertex
        PetscInt *star = nullptr;
        PetscInt numStar;
        DMPlexGetTransitiveClosure(dm, v, PETSC_FALSE, &numStar, &star);
        DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);

        PetscInt cell;
        stencilData.stencil[10];
        // extract the connected cells and store them
        for (cl = 0; cl < numStar * 2; cl += 2) {
            cell = star[cl];
            if (cell < cStart || cell >= cEnd) continue;

            stencilData.stencil.push_back(cell);
            // extract and save the cell info
            PetscFVCellGeom *cg;
            DMPlexPointLocalRead(dmCell, cell, cellGeomArray, &cg);

            for (PetscInt d = 0; d < dim; ++d) {
                // add up distance of cell centers to the vertex and store
                stencilData.gradientWeights[d] += abs(cg->centroid[d] - xyz[d]);
            }
            stencilData.stencilSize += 1;
        }
        DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &numStar, &star);
        // store the stencils of this vertex
        vertexStencils.push_back(std::move(stencilData));
    }

    vertexStencils.data();
    PetscScalar *eulerSource = nullptr;
    Vec computedF;
    PetscScalar *sourceArray;
    PetscScalar *solution;
    VecGetArray(domain->GetSolutionVector(), &solution);

    // copy over euler
    PetscScalar *eulerField = nullptr;
    DMPlexPointLocalFieldRef(domain->GetDM(), GetParam().cellNumber, domain->GetField("euler").id, solution, &eulerField);
    // copy over euler
    for (std::size_t i = 0; i < GetParam().inputEulerValues.size(); i++) {
        eulerField[i] = GetParam().inputEulerValues[i];
    }

    VecRestoreArray(domain->GetSolutionVector(), &solution);

    auto process = ablate::finiteVolume::processes::SurfaceForce(GetParam().sigma);
    process.vertexStencils = vertexStencils;
    DMGetLocalVector(domain->GetDM(), &computedF);
    VecZeroEntries(computedF);

    ablate::finiteVolume::processes::SurfaceForce::ComputeSource(*fvSolver, domain->GetDM(), NULL, domain->GetSolutionVector(), computedF, &process);

    // ASSERT
    VecGetArray(computedF, &sourceArray);

    DMPlexPointLocalFieldRef(domain->GetDM(), GetParam().cellNumber, domain->GetField("euler").id, sourceArray, &eulerSource);
    for (std::size_t c = 0; c < GetParam().expectedEulerSource.size(); c++) {
        ASSERT_LT(PetscAbs((GetParam().expectedEulerSource[c] - eulerSource[c]) / (GetParam().expectedEulerSource[c] + 1E-30)), params.errorTolerance)
            << "The percent difference for the expected and actual source (" << GetParam().expectedEulerSource[c] << " vs " << eulerSource[c] << ") should be small for index " << c;
    }
    VecRestoreArray(computedF, &sourceArray) >> ablate::utilities::PetscUtilities::checkError;

    DMRestoreLocalVector(domain->GetDM(), &computedF) >> ablate::utilities::PetscUtilities::checkError;
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
                                                                      .inputVFfield = ablate::mathFunctions::Create(" 1.0"),
                                                                      .expectedEulerSource = {0, 0, 0}},
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

                                         ));