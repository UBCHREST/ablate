#include "surfaceForce.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"

ablate::finiteVolume::processes::SurfaceForce::SurfaceForce(PetscReal sigma) : sigma(sigma) {}

void ablate::finiteVolume::processes::SurfaceForce::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    /** Make stencils for connected cells of each vertex and store them
     * extract the vortices and get their coordinates
     * march over each vertex and store the vertex point
     * extract the cells in the domain then identify the connected cells to the vertex using "PETSc-Closure" and store
     * extract the connected cells info and store
     * calculate the weights for gradient by summing the distances of connected cells to the vertex and store
     * push back for this vertex
     **/
    auto dim = flow.GetSubDomain().GetDimensions();
    auto dm = flow.GetSubDomain().GetDM();
    Vec cellGeomVec;
    DM dmCell;
    const PetscScalar *cellGeomArray;
    DMPlexGetGeometryFVM(dm, nullptr, &cellGeomVec, nullptr) >> utilities::PetscUtilities::checkError;
    VecGetDM(cellGeomVec, &dmCell);
    VecGetArrayRead(cellGeomVec, &cellGeomArray);
    // create a domain function, dmData, to use it in main function for storing any calculated data/field. Here the vertex normals will be stored on vortices, therefore k = 1
    PetscFE fe_coords;
    PetscInt k = 1;
    DMClone(dm, &dmData) >> utilities::PetscUtilities::checkError;
    PetscFECreateLagrange(PETSC_COMM_SELF, dim, dim, PETSC_TRUE, k, PETSC_DETERMINE, &fe_coords) >> utilities::PetscUtilities::checkError;
    DMSetField(dmData, 0, NULL, (PetscObject)fe_coords) >> utilities::PetscUtilities::checkError;
    DMCreateDS(dmData) >> utilities::PetscUtilities::checkError;
    // extract the local coordinates array
    Vec localCoordsVector;
    PetscSection coordsSection;
    PetscScalar *coordsArray;
    DMGetCoordinateSection(dm, &coordsSection) >> utilities::PetscUtilities::checkError;
    DMGetCoordinatesLocal(dm, &localCoordsVector) >> utilities::PetscUtilities::checkError;
    VecGetArray(localCoordsVector, &coordsArray) >> utilities::PetscUtilities::checkError;

    PetscInt vStart, vEnd, cStart, cEnd, cl;
    // extract vortices of domain
    DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd) >> utilities::PetscUtilities::checkError;

    // march over vortices
    for (PetscInt v = vStart; v < vEnd; v++) {
        auto newStencil = VertexStencil{};
        // store the vertex point
        newStencil.vertexId = v;
        newStencil.stencilCoord = {0, 0, 0};
        PetscInt off;
        PetscReal xyz[3];
        // extract x, y, z values
        PetscSectionGetOffset(coordsSection, v, &off) >> utilities::PetscUtilities::checkError;
        for (PetscInt d = 0; d < dim; ++d) {
            xyz[d] = coordsArray[off + d];
            // store coordinates of the vertex
            newStencil.stencilCoord[d] = xyz[d];
        }
        newStencil.stencil[10];
        newStencil.gradientWeights = {0, 0, 0};
        newStencil.stencilSize = 0;
        // extract nodes of the vertex
        PetscInt *star = nullptr;
        PetscInt numStar;
        DMPlexGetTransitiveClosure(dm, v, PETSC_FALSE, &numStar, &star) >> utilities::PetscUtilities::checkError;

        // extract the connected cells and store them
        DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;
        PetscInt cell;
        for (cl = 0; cl < numStar * 2; cl += 2) {
            cell = star[cl];
            if (cell < cStart || cell >= cEnd) continue;
            newStencil.stencil.push_back(cell);
            PetscFVCellGeom *cg;
            DMPlexPointLocalRead(dmCell, cell, cellGeomArray, &cg) >> utilities::PetscUtilities::checkError;

            for (PetscInt d = 0; d < dim; ++d) {
                // add up distance of cell centers to the vertex and store
                newStencil.gradientWeights[d] += abs(cg->centroid[d] - xyz[d] + utilities::Constants::tiny);
            }
            newStencil.stencilSize += 1;
        }
        DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &numStar, &star) >> utilities::PetscUtilities::checkError;
        // store the stencils of this vertex
        vertexStencils.push_back(std::move(newStencil));
    }
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArray(localCoordsVector, &coordsArray) >> utilities::PetscUtilities::checkError;

    flow.RegisterRHSFunction(ComputeSource, this);
}

PetscErrorCode ablate::finiteVolume::processes::SurfaceForce::ComputeSource(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx) {
    PetscFunctionBegin;

    /** Now use the stored vertex information to calculate curvature at each cell center
     * march over the stored vortices to read the alpha values of connected cells
     * calculate the normal at each vertex using alpha values of it's cells
     * march over cells in the domain
     * extract connected vortices to each cell using "closure" and read the saved normals of vortices
     * calculate the normal at the cell center
     * calculate the gradient of magnitude of vertex normals and divergent of normals at the cell center
     * use the computed values to calculate the curvature at the center
     **/

    auto process = (ablate::finiteVolume::processes::SurfaceForce *)ctx;
    auto fields = solver.GetSubDomain().GetFields();

    // Look for the euler field and volume fraction (alpha)
    const auto &eulerField = solver.GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);
    const auto &VFfield = solver.GetSubDomain().GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);

    // get the cell range
    solver::Range cellRange;
    solver.GetCellRangeWithoutGhost(cellRange);
    PetscScalar *fArray;
    VecGetArray(locFVec, &fArray) >> utilities::PetscUtilities::checkError;

    auto dim = solver.GetSubDomain().GetDimensions();

    // get the flowSolution
    const PetscScalar *solArray;
    VecGetArrayRead(locX, &solArray) >> utilities::PetscUtilities::checkError;

    // Get the cell geometry
    Vec cellGeomVec;
    DM dmCell;
    const PetscScalar *cellGeomArray;
    DMPlexGetGeometryFVM(dm, nullptr, &cellGeomVec, nullptr) >> utilities::PetscUtilities::checkError;
    VecGetDM(cellGeomVec, &dmCell) >> utilities::PetscUtilities::checkError;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;

    // get the coordinate domain
    DM cdm;
    DMGetCoordinateDM(dm, &cdm) >> utilities::PetscUtilities::checkError;
    // create a domain to get a local vector to save normals
    Vec localVec;
    DMGetLocalVector(process->dmData, &localVec) >> utilities::PetscUtilities::checkError;
    PetscScalar *normalArray = NULL;
    VecGetArray(localVec, &normalArray);
    PetscScalar *vertexNormal;
    // extract the local coordinates array
    Vec localCoordsVector;
    PetscSection coordsSection;
    PetscScalar *coordsArray = NULL;
    DMGetCoordinateSection(dm, &coordsSection) >> utilities::PetscUtilities::checkError;
    DMGetCoordinatesLocal(dm, &localCoordsVector) >> utilities::PetscUtilities::checkError;
    VecGetArray(localCoordsVector, &coordsArray) >> utilities::PetscUtilities::checkError;

    // march over the stored vortices
    for (const auto &info : process->vertexStencils) {
        PetscReal totalAlpha[3] = {0, 0, 0};
        // march over the connected cells to each vertex and get the cell info and filed value
        for (PetscInt p = 0; p < info.stencilSize; p++) {
            PetscFVCellGeom *cg = nullptr;
            DMPlexPointLocalRead(dmCell, info.stencil[p], cellGeomArray, &cg) >> utilities::PetscUtilities::checkError;

            PetscScalar *alpha = nullptr;
            DMPlexPointLocalFieldRead(dm, info.stencil[p], VFfield.id, solArray, &alpha) >> utilities::PetscUtilities::checkError;

            // add up the contribution of the cell. Front cells have positive and back cells have negative contribution to the vertex normal
            PetscReal alphaVal[dim];
            for (PetscInt d = 0; d < dim; ++d) {
                if (cg->centroid[d] > info.stencilCoord[d]) {
                    alphaVal[d] = *alpha;

                } else if (cg->centroid[d] < info.stencilCoord[d]) {
                    alphaVal[d] = -*alpha;
                }
                totalAlpha[d] += alphaVal[d];
            }
        }
        // calculate and save the normal of the vertex
        DMPlexPointLocalRef(cdm, info.vertexId, normalArray, &vertexNormal) >> utilities::PetscUtilities::checkError;
        for (PetscInt d = 0; d < dim; ++d) {
            vertexNormal[d] = totalAlpha[d] / info.gradientWeights[d];
        }
    }
    // march over cells
    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
        const PetscInt c = cellRange.points ? cellRange.points[i] : i;
        // get current euler solution here to get velocity
        const PetscScalar *euler = nullptr;
        DMPlexPointLocalFieldRead(dm, c, eulerField.id, solArray, &euler) >> utilities::PetscUtilities::checkError;
        auto density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];

        PetscScalar vel[dim];
        for (PetscInt d = 0; d < dim; d++) {
            vel[d] = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density;
        }

        // add calculated sources to euler
        PetscScalar *eulerSource = nullptr;
        DMPlexPointLocalFieldRef(dm, c, eulerField.id, fArray, &eulerSource) >> utilities::PetscUtilities::checkError;

        PetscReal surfaceEnergy = 0;
        PetscReal totalGradMagNormal = 0;
        PetscReal divergentNormal[3] = {0, 0, 0};
        PetscReal grad[3] = {0, 0, 0};
        PetscReal gradMagNormal[dim];
        PetscReal curvature;
        PetscScalar surfaceForce[dim];

        // get the centroid information for the cell
        PetscFVCellGeom *fcg;
        DMPlexPointLocalRead(dmCell, c, cellGeomArray, &fcg) >> utilities::PetscUtilities::checkError;

        // extract connected vortices to the cell
        PetscInt *closure = NULL;
        PetscInt numClosure;
        DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &numClosure, &closure) >> utilities::PetscUtilities::checkError;
        PetscInt cl, numVertex = 0;

        PetscReal centerNormal[3] = {0, 0, 0};
        PetscReal gradNormal[dim];
        PetscReal cellCenterNormal[dim];
        PetscReal totalDivNormal = 0;
        PetscInt vStart, vEnd;
        DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd) >> utilities::PetscUtilities::checkError;
        PetscReal distance[3] = {0, 0, 0};
        PetscReal xyz[3];
        PetscInt offset;

        for (cl = 0; cl < numClosure * 2; cl += 2) {
            PetscInt vertex = closure[cl];

            if (vertex < vStart || vertex >= vEnd) continue;
            // sum up the number of connected vortices
            numVertex += 1;
            PetscSectionGetOffset(coordsSection, vertex, &offset) >> utilities::PetscUtilities::checkError;

            for (PetscInt d = 0; d < dim; ++d) {
                // extract coordinates of vortices of this cell and calculate the distance to the center
                DMPlexPointLocalRead(cdm, vertex, normalArray, &vertexNormal) >> utilities::PetscUtilities::checkError;
                xyz[d] = coordsArray[offset + d];

                // calculate normal at each cell center using normals of connected vortices to the cell
                centerNormal[d] += vertexNormal[d];

                // calculate divergence of normal for each cell center using vertex normals --> Delta.ni,j=  [nx(i+1/2,j+1/2)-nx(i-1/2,j+1/2)+nx(i+1/2,j-1/2)-nx(i-1/2,j-1/2)]/2*deltaX +
                // [ny(i+1/2,j+1/2)-ny(i+1/2,j-1/2)+ny(i-1/2,j+1/2)-ny(i-1/2,j-1/2)]/2*deltaY
                // get magnitude of normals at vortices to calculate the derivative of the magnitude of normals at the cell center
                distance[d] += abs(xyz[d] - fcg->centroid[d] + utilities::Constants::tiny);
                const PetscReal magVertexNormal = utilities::MathUtilities::MagVector(dim, vertexNormal);
                if (fcg->centroid[d] < xyz[d]) {
                    divergentNormal[d] += vertexNormal[d];
                    grad[d] += magVertexNormal;

                } else if (fcg->centroid[d] > xyz[d]) {
                    divergentNormal[d] -= vertexNormal[d];
                    grad[d] -= magVertexNormal;
                }
            }
        }
        for (PetscInt d = 0; d < dim; ++d) {
            totalDivNormal += divergentNormal[d] / distance[d];
            gradNormal[d] = grad[d] / distance[d];
            cellCenterNormal[d] = centerNormal[d] / numVertex;
        }
        // magnitude of normal at the center
        const PetscReal magCellNormal = utilities::MathUtilities::MagVector(dim, cellCenterNormal);
        for (PetscInt d = 0; d < dim; ++d) {
            // (ni,j/ |ni,j| . Delta)
            gradMagNormal[d] = (cellCenterNormal[d] / (magCellNormal + utilities::Constants::tiny)) * gradNormal[d];
            totalGradMagNormal += gradMagNormal[d];
        }
        // calculate curvature -->  kappa = 1/n [(n/|n|. Delta) |n| - (Delta.n)]
        curvature = (totalGradMagNormal - totalDivNormal) / (magCellNormal + utilities::Constants::tiny);

        for (PetscInt d = 0; d < dim; ++d) {
            // calculate surface force and energy
            surfaceForce[d] = process->sigma * curvature * cellCenterNormal[d];
            surfaceEnergy += surfaceForce[d] * vel[d];
            // add in the contributions
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = surfaceForce[d];
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] = surfaceEnergy;
        }
        DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &numClosure, &closure) >> utilities::PetscUtilities::checkError;
    }
    // cleanup
    solver.RestoreRange(cellRange);
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArray(locFVec, &fArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArray(localCoordsVector, &coordsArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(locX, &solArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArray(localVec, &normalArray);
    DMRestoreLocalVector(process->dmData, &localVec);

    PetscFunctionReturn(0);
}

ablate::finiteVolume::processes::SurfaceForce::~SurfaceForce() { DMDestroy(&dmData); }

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::SurfaceForce, "calculates surface tension force and adds source terms",
         ARG(PetscReal, "sigma", "sigma, surface tension coefficient"));
