#include "domain/RBF/phs.hpp"
#include "surfaceForce.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"

ablate::finiteVolume::processes::SurfaceForce::SurfaceForce(PetscReal sigma) : sigma(sigma) {}

// Done once at the beginning of every run
void ablate::finiteVolume::processes::SurfaceForce::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {


    PetscInt polyAug = 8;
    PetscInt phsOrder = 2;

    // Create the radial basis function. This can be changed for another one but make sure to change the include above
    auto rbf = std::make_shared<ablate::domain::rbf::PHS>(polyAug, phsOrder, false, false);
    rbf->Setup(flow.GetSubDomain());

    flow.RegisterRHSFunction(ComputeSource, this);
}

// Called every time the mesh changes
void ablate::finiteVolume::processes::SurfaceForce::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
  rbf->Initialize();
}

PetscErrorCode ablate::finiteVolume::processes::SurfaceForce::ComputeSource(const FiniteVolumeSolver &flow, DM dm, PetscReal time, Vec locX, Vec locF, void *ctx) {
    PetscFunctionBegin;

//    auto process = (ablate::finiteVolume::processes::SurfaceForce *)ctx;
//    const auto subDomain = solver.GetSubDomain();
//    const PetscInt dim = subDomain.GetDimensions();


//    // Look for the euler field and volume fraction (alpha)
//    const ablate::domain::Field *eulerField = subDomain.GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);
//    const ablate::domain::Field *vofField = subDomain.GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);

//    // Get the DMs of the two fields. Note that each is obtained in case they change from SOL to AUX at some point in the future.
//    // Note that the physical layout of the two DMs will be the same, but which DM the VECs are stored in
//    DM vofDM = subDomain.GetFieldDM(vofField);
//    DM eulerDM = subDomain.GetFieldDM(velField);

//    ablate::domain::Range cellRange, vertRange;
//    solver.GetCellRangeWithoutGhost(&cellRange);
//    solver.GetRange(0, vertRange);

//    // Create a work array to store the unit normal at cell vertices
//    PetscReal *n;
//    DMGetWorkArray(vofDM, dim*(vertRange.end - vertRange.start), MPIU_REAL, &n) >> utilities::PetscUtilities::checkError;

//    // Shift it so that we can access elements directly
//    n -= vertRange.start;

//    const PetscScalar *xArray;
//    VecGetArrayRead(locX, &xArray) >> utilities::PetscUtilities::checkError;

//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//      PetscInt cell = cellRange.GetPoint(c);

//      const PetscReal *vofVal;
//      xDMPlexPointLocalRead(vofDM, cell, vofFIeld->id, &vofVal);

//      // Only worry about cut-cells
//      if ( ((*vofVal) > ablate::utilities::Constants::small) && ((*vofVal) < (1.0 - ablate::utilities::Constants::small)) ) {

//      }

//    }



//    n += vertRange.start; // If you don't shift it back the incorrect memory block will be released.
//    DMRestoreWorkArray(vofDM, 0, MPIU_REAL, &n) >> utilities::PetscUtilities::checkError;
//    solver.RestoreRange(&cellRange);
//    solver.RestoreRange(&vertRange);

























//    // get the cell range
//    ablate::domain::Range cellRange;
//    solver.GetCellRangeWithoutGhost(cellRange);
//    PetscScalar *fArray;
//    VecGetArray(locFVec, &fArray) >> utilities::PetscUtilities::checkError;

//    auto dim = subDomain.GetDimensions();

//    // get the flowSolution
//    const PetscScalar *solArray;
//    VecGetArrayRead(locX, &solArray) >> utilities::PetscUtilities::checkError;



















//    // march over the stored vortices
//    for (const auto &info : process->vertexStencils) {
//        PetscReal totalAlpha[3] = {0, 0, 0};
//        // march over the connected cells to each vertex and get the cell info and filed value
//        for (PetscInt p = 0; p < info.stencilSize; p++) {
//            PetscFVCellGeom *cg = nullptr;
//            DMPlexPointLocalRead(dmCell, info.stencil[p], cellGeomArray, &cg) >> utilities::PetscUtilities::checkError;

//            const PetscScalar *alpha = nullptr;
//            DMPlexPointLocalFieldRead(dm, info.stencil[p], velField.id, solArray, &alpha) >> utilities::PetscUtilities::checkError;

//            // add up the contribution of the cell. Front cells have positive and back cells have negative contribution to the vertex normal
//            PetscReal alphaVal[3];
//            if (alpha) {
//                for (PetscInt d = 0; d < dim; ++d) {
//                    if (cg->centroid[d] > info.stencilCoord[d]) {
//                        alphaVal[d] = alpha[0];

//                    } else if (cg->centroid[d] < info.stencilCoord[d]) {
//                        alphaVal[d] = -alpha[0];
//                    }
//                    totalAlpha[d] += alphaVal[d];
//                }
//            }
//        }
//        // calculate and save the normal of the vertex
//        DMPlexPointLocalRef(cdm, info.vertexId, normalArray, &vertexNormal) >> utilities::PetscUtilities::checkError;
//        for (PetscInt d = 0; d < dim; ++d) {
//            vertexNormal[d] = totalAlpha[d] / info.gradientWeights[d];
//        }
//    }
//    // march over cells
//    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
//        const PetscInt c = cellRange.points ? cellRange.points[i] : i;
//        // get current euler solution here to get velocity
//        const PetscScalar *euler = nullptr;
//        DMPlexPointLocalFieldRead(dm, c, eulerField.id, solArray, &euler) >> utilities::PetscUtilities::checkError;
//        auto density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];

//        PetscScalar vel[dim];
//        for (PetscInt d = 0; d < dim; d++) {
//            vel[d] = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density;
//        }

//        // add calculated sources to euler
//        PetscScalar *eulerSource = nullptr;
//        DMPlexPointLocalFieldRef(dm, c, eulerField.id, fArray, &eulerSource) >> utilities::PetscUtilities::checkError;

//        PetscReal surfaceEnergy = 0;
//        PetscReal totalGradMagNormal = 0;
//        PetscReal divergentNormal[3] = {0, 0, 0};
//        PetscReal grad[3] = {0, 0, 0};
//        PetscReal gradMagNormal[dim];
//        PetscReal curvature;
//        PetscScalar surfaceForce[dim];

//        // get the centroid information for the cell
//        PetscFVCellGeom *fcg;
//        DMPlexPointLocalRead(dmCell, c, cellGeomArray, &fcg) >> utilities::PetscUtilities::checkError;

//        // extract connected vortices to the cell
//        PetscInt *closure = NULL;
//        PetscInt numClosure;
//        DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &numClosure, &closure) >> utilities::PetscUtilities::checkError;
//        PetscInt cl, numVertex = 0;

//        PetscReal centerNormal[3] = {0, 0, 0};
//        PetscReal gradNormal[dim];
//        PetscReal cellCenterNormal[dim];
//        PetscReal totalDivNormal = 0;
//        PetscInt vStart, vEnd;
//        DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd) >> utilities::PetscUtilities::checkError;
//        PetscReal distance[3] = {0, 0, 0};
//        PetscReal xyz[3];
//        PetscInt offset;

//        for (cl = 0; cl < numClosure * 2; cl += 2) {
//            PetscInt vertex = closure[cl];

//            if (vertex < vStart || vertex >= vEnd) continue;
//            // sum up the number of connected vortices
//            numVertex += 1;
//            PetscSectionGetOffset(coordsSection, vertex, &offset) >> utilities::PetscUtilities::checkError;

//            for (PetscInt d = 0; d < dim; ++d) {
//                // extract coordinates of vortices of this cell and calculate the distance to the center
//                DMPlexPointLocalRead(cdm, vertex, normalArray, &vertexNormal) >> utilities::PetscUtilities::checkError;
//                xyz[d] = coordsArray[offset + d];

//                // calculate normal at each cell center using normals of connected vortices to the cell
//                centerNormal[d] += vertexNormal[d];

//                // calculate divergence of normal for each cell center using vertex normals --> Delta.ni,j=  [nx(i+1/2,j+1/2)-nx(i-1/2,j+1/2)+nx(i+1/2,j-1/2)-nx(i-1/2,j-1/2)]/2*deltaX +
//                // [ny(i+1/2,j+1/2)-ny(i+1/2,j-1/2)+ny(i-1/2,j+1/2)-ny(i-1/2,j-1/2)]/2*deltaY
//                // get magnitude of normals at vortices to calculate the derivative of the magnitude of normals at the cell center
//                distance[d] += abs(xyz[d] - fcg->centroid[d] + utilities::Constants::tiny);
//                const PetscReal magVertexNormal = utilities::MathUtilities::MagVector(dim, vertexNormal);
//                if (fcg->centroid[d] < xyz[d]) {
//                    divergentNormal[d] += vertexNormal[d];
//                    grad[d] += magVertexNormal;

//                } else if (fcg->centroid[d] > xyz[d]) {
//                    divergentNormal[d] -= vertexNormal[d];
//                    grad[d] -= magVertexNormal;
//                }
//            }
//        }
//        for (PetscInt d = 0; d < dim; ++d) {
//            totalDivNormal += divergentNormal[d] / distance[d];
//            gradNormal[d] = grad[d] / distance[d];
//            cellCenterNormal[d] = centerNormal[d] / numVertex;
//        }
//        // magnitude of normal at the center
//        const PetscReal magCellNormal = utilities::MathUtilities::MagVector(dim, cellCenterNormal);
//        for (PetscInt d = 0; d < dim; ++d) {
//            // (ni,j/ |ni,j| . Delta)
//            gradMagNormal[d] = (cellCenterNormal[d] / (magCellNormal + utilities::Constants::tiny)) * gradNormal[d];
//            totalGradMagNormal += gradMagNormal[d];
//        }
//        // calculate curvature -->  kappa = 1/n [(n/|n|. Delta) |n| - (Delta.n)]
//        curvature = (totalGradMagNormal - totalDivNormal) / (magCellNormal + utilities::Constants::tiny);

//        for (PetscInt d = 0; d < dim; ++d) {
//            // calculate surface force and energy
//            surfaceForce[d] = process->sigma * curvature * cellCenterNormal[d];
//            surfaceEnergy += surfaceForce[d] * vel[d];
//            // add in the contributions
//            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = surfaceForce[d];
//            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] = surfaceEnergy;
//        }
//        DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &numClosure, &closure) >> utilities::PetscUtilities::checkError;
//    }
//    // cleanup
//    solver.RestoreRange(cellRange);
//    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
//    VecRestoreArray(locFVec, &fArray) >> utilities::PetscUtilities::checkError;
//    VecRestoreArray(localCoordsVector, &coordsArray) >> utilities::PetscUtilities::checkError;
//    VecRestoreArrayRead(locX, &solArray) >> utilities::PetscUtilities::checkError;
//    VecRestoreArray(localVec, &normalArray);
//    DMRestoreLocalVector(process->dmData, &localVec);

    PetscFunctionReturn(PETSC_SUCCESS);
}

ablate::finiteVolume::processes::SurfaceForce::~SurfaceForce() { DMDestroy(&dmData); }

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::SurfaceForce, "calculates surface tension force and adds source terms",
         ARG(PetscReal, "sigma", "sigma, surface tension coefficient"));
