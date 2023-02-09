#include "surfaceForce.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/mathUtilities.hpp"

ablate::finiteVolume::processes::SurfaceForce::SurfaceForce(PetscReal sigma) : sigma(sigma) {}

void ablate::finiteVolume::processes::SurfaceForce::Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) { flow.RegisterRHSFunction(ComputeSource, this); }

PetscErrorCode ablate::finiteVolume::processes::SurfaceForce::ComputeSource(const FiniteVolumeSolver& solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void* ctx) {
    PetscFunctionBegin;
    auto process = (ablate::finiteVolume::processes::SurfaceForce*)ctx;
    auto fields = solver.GetSubDomain().GetFields();

    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto& field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    auto VFfield = std::find_if(fields.begin(), fields.end(), [](const auto& field) { return field.name == TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD; });

    // get the cell range
    solver::Range cellRange;
    solver.GetCellRangeWithoutGhost(cellRange);

    PetscScalar* fArray;
    VecGetArray(locFVec, &fArray);

    auto dim = solver.GetSubDomain().GetDimensions();

    // get the flowSolution
    const PetscScalar* solArray;
    VecGetArrayRead(locX, &solArray);
    // Get the cell geometry
    Vec cellGeomVec;
    DM dmCell;
    const PetscScalar* cellGeomArray;
    DMPlexGetGeometryFVM(dm, nullptr, &cellGeomVec, nullptr);
    VecGetDM(cellGeomVec, &dmCell);
    VecGetArrayRead(cellGeomVec, &cellGeomArray);

    // march over cells
    for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
        const PetscInt c = cellRange.points ? cellRange.points[i] : i;

        // get the centroid information for the cell
        PetscFVCellGeom* fcg;
        DMPlexPointLocalRead(dmCell, c, cellGeomArray, &fcg);

        PetscReal surfaceEnergy = 0;
        PetscReal totalGradMagNormal = 0;
        PetscReal TINY = 1e-30;
        PetscReal numVertex;
        PetscReal numVortices = 0;

        PetscReal divergentNormal[dim];
        PetscReal grad[dim];
        PetscReal gradMagNormal[dim];
        PetscReal curvature;
        PetscScalar surfaceForce[dim];

        // get current euler solution here
        const PetscScalar* euler = nullptr;
        DMPlexPointLocalFieldRead(dm, c, eulerField->id, solArray, &euler);
        auto density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];

        PetscScalar vel[dim];
        for (PetscInt d = 0; d < dim; d++) {
            vel[d] = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density;
        }
        // add calculated sources to euler
        PetscScalar* eulerSource = nullptr;
        DMPlexPointLocalFieldRef(dm, c, eulerField->id, fArray, &eulerSource);

        PetscReal centerDivNormal = 0;
        PetscReal cellCenterNormal[3] = {0, 0, 0};
        PetscReal totGradNormal[3] = {0, 0, 0};

        // check to see if there is a ghost label
        DMLabel ghostLabel;
        PetscBool boundary;
        DMGetLabel(dm, "ghost", &ghostLabel);
        PetscInt ghost = -1;
        DMIsBoundaryPoint(dm, c, &boundary);
        if (ghostLabel) {
            DMLabelGetValue(ghostLabel, c, &ghost);
        }
        if (ghost >= 0 || boundary) continue;
        // get faces of this cell
        const PetscInt* cellFaces;
        PetscInt numCellFaces;
        DMPlexGetCone(dm, c, &cellFaces);
        DMPlexGetConeSize(dm, c, &numCellFaces);
        // get number of vortices of the cell
        numVertex = numCellFaces;

        // get vortices attached to each face
        for (PetscInt f = 0; f < numCellFaces; f++) {
            // here if the case is 2D extract vortices, if 3D extract edges then proceed
            if (dim == 2) {
                const PetscInt* attachVortices;
                PetscInt numAttachVortices;
                DMPlexGetCone(dm, cellFaces[f], &attachVortices);
                DMPlexGetConeSize(dm, cellFaces[f], &numAttachVortices);

                // zero out
                PetscReal normal[3] = {0, 0, 0};
                PetscReal gradNormal[3] = {0, 0, 0};
                PetscReal totalDivNormal = 0;
                PetscReal centerNormal[3] = {0, 0, 0};

                Vec localCoordsVector;
                PetscSection coordsSection;
                PetscScalar* coordsArray;
                DMGetCoordinateSection(dm, &coordsSection);
                DMGetCoordinatesLocal(dm, &localCoordsVector);
                VecGetArray(localCoordsVector, &coordsArray);
                // get coordinates of vortices of this cell
                for (PetscInt v = 0; v < numAttachVortices; ++v) {
                    PetscInt off;
                    PetscReal xyz[3];
                    // extract x, y, z values
                    PetscSectionGetOffset(coordsSection, attachVortices[v], &off);
                    for (PetscInt d = 0; d < dim; ++d) {
                        xyz[d] = coordsArray[off + d];
                    }

                    // get faces attached to each vertex
                    const PetscInt* attachedFace;
                    PetscInt numAttachedFace;

                    DMPlexGetSupport(dm, attachVortices[v], &attachedFace);
                    DMPlexGetSupportSize(dm, attachVortices[v], &numAttachedFace);

                    PetscReal totAlpha[3] = {0, 0, 0};
                    PetscReal distCell[3] = {0, 0, 0};

                    // exclude vortices that do not have enough faces
                    if (numAttachedFace < 2 * dim) {
                        totAlpha[3] = 0;
                    }

                    else if (numAttachedFace >= 2 * dim) {
                        // Get cells attached to each face
                        for (PetscInt iFace = 0; iFace < numAttachedFace; ++iFace) {
                            const PetscInt* cell;
                            PetscInt numCell;

                            DMPlexGetSupport(dm, attachedFace[iFace], &cell);
                            DMPlexGetSupportSize(dm, attachedFace[iFace], &numCell);
                            if (numCell != 2) continue;

                            PetscReal vertexAlpha[3] = {0, 0, 0};
                            PetscReal distance[3] = {0, 0, 0};

                            // march over attached-cells of each vertex to calculate normal
                            for (PetscInt iCell = 0; iCell < numCell; ++iCell) {
                                PetscReal verNormal[3] = {0, 0, 0};
                                // get centroid for attached-cells to each vertex
                                PetscFVCellGeom* cg;
                                DMPlexPointLocalRead(dmCell, cell[iCell], cellGeomArray, &cg);
                                // get current alpha values of attached cells
                                PetscScalar* alpha = nullptr;
                                DMPlexPointLocalFieldRead(dm, cell[iCell], VFfield->id, solArray, &alpha);

                                // calculate normal at each vertex using alpha -->   n(i+1/2, j+1/2) = x[alpha(i+1,j)-alpha(i,j)+alpha(i+1,j+1)-alpha(i,j+1)]/2*deltaX +
                                // y[alpha(i,j+1)-alpha(i,j)+alpha(i+1,j+1)-alpha(i+1,j)]/2*deltaY
                                for (PetscInt d = 0; d < dim; ++d) {
                                    if (cg->centroid[d] > xyz[d]) {
                                        verNormal[d] = *alpha;

                                    } else if (cg->centroid[d] < xyz[d]) {
                                        verNormal[d] = -*alpha;

                                    } else {
                                        verNormal[d] = 0;
                                    }
                                    // add up distance of centers to the vertex
                                    distance[d] += abs(cg->centroid[d] - xyz[d] + TINY);
                                    vertexAlpha[d] += verNormal[d];
                                }
                            }
                            for (PetscInt d = 0; d < dim; ++d) {
                                // add up the contributions of vortices of faces
                                distCell[d] += distance[d];
                                totAlpha[d] += vertexAlpha[d];
                            }
                        }
                    }
                    // Now calculate normal at each cell center using normals of attached vortices
                    for (PetscInt d = 0; d < dim; ++d) {
                        // divide by "dim" since each attached cell is used "dim" times
                        normal[d] = totAlpha[d] / (distCell[d] + TINY) / dim;

                        // calculate divergence of normal for each cell center using vertex normals --> Delta.ni,j=  [nx(i+1/2,j+1/2)-nx(i-1/2,j+1/2)+nx(i+1/2,j-1/2)-nx(i-1/2,j-1/2)]/2*deltaX +
                        // [ny(i+1/2,j+1/2)-ny(i+1/2,j-1/2)+ny(i-1/2,j+1/2)-ny(i-1/2,j-1/2)]/2*deltaY
                        if (fcg->centroid[d] < xyz[d] || fcg->centroid[d] > xyz[d]) {
                            divergentNormal[d] = normal[d] / (xyz[d] - fcg->centroid[d]);

                        } else {
                            divergentNormal[d] = 0;
                        }
                        totalDivNormal += divergentNormal[d] / numVertex;

                        // calculate normal at each cell center using normals of attached vortices to the cell
                        centerNormal[d] += normal[d] / numVertex;

                        // calculate magnitude of normals at vortices
                        const PetscReal magVertexNormal = utilities::MathUtilities::MagVector(dim, normal);
                        // calculate the derivative of the magnitude of normals at the cell center
                        if (fcg->centroid[d] < xyz[d] || fcg->centroid[d] > xyz[d]) {
                            grad[d] = magVertexNormal / (xyz[d] - fcg->centroid[d]);

                        } else {
                            grad[d] = 0;
                        }
                        gradNormal[d] += grad[d] / numVertex;
                    }
                }
                VecRestoreArray(localCoordsVector, &coordsArray);
                for (PetscInt d = 0; d < dim; ++d) {
                    // divide by "dim" since  each vertex is used "dim" times
                    totGradNormal[d] += gradNormal[d] / dim;
                    cellCenterNormal[d] += centerNormal[d] / dim;
                    centerDivNormal += totalDivNormal / dim;
                }
            }
            if (dim == 3) {
                const PetscInt* attachEdges;
                PetscInt numAttachEdges;
                // find edges attached to faces of the cell
                DMPlexGetCone(dm, cellFaces[f], &attachEdges);
                DMPlexGetConeSize(dm, cellFaces[f], &numAttachEdges);

                // calculate the number of vortices of the cell
                for (PetscInt face = 0; face < numCellFaces; face++) {
                    for (PetscInt e = 0; e < numAttachEdges; e++) {
                        numVortices += numAttachEdges / dim;
                    }
                }
                for (PetscInt e = 0; e < numAttachEdges; e++) {
                    const PetscInt* attachVortices;
                    PetscInt numAttachVortices;
                    // find vortices attached to this edge
                    DMPlexGetCone(dm, attachEdges[e], &attachVortices);
                    DMPlexGetConeSize(dm, attachEdges[e], &numAttachVortices);

                    PetscReal normal[3] = {0, 0, 0};
                    PetscReal gradNormal[3] = {0, 0, 0};
                    PetscReal totalDivNormal = 0;
                    PetscReal centerNormal[3] = {0, 0, 0};

                    Vec localCoordsVector;
                    PetscSection coordsSection;
                    PetscScalar* coordsArray;
                    DMGetCoordinateSection(dm, &coordsSection);
                    DMGetCoordinatesLocal(dm, &localCoordsVector);
                    VecGetArray(localCoordsVector, &coordsArray);

                    // get coordinates of vortices of this cell
                    for (PetscInt v = 0; v < numAttachVortices; ++v) {
                        PetscInt off;
                        PetscReal xyz[3];
                        // extract x, y, z values
                        PetscSectionGetOffset(coordsSection, attachVortices[v], &off);

                        for (PetscInt d = 0; d < dim; ++d) {
                            xyz[d] = coordsArray[off + d];
                        }
                        PetscReal totAlpha[3] = {0, 0, 0};
                        PetscReal distCell[3] = {0, 0, 0};
                        // get faces attached to each vertex
                        const PetscInt* edges;
                        PetscInt numEdges;

                        DMPlexGetSupport(dm, attachVortices[v], &edges);
                        DMPlexGetSupportSize(dm, attachVortices[v], &numEdges);

                        if (numEdges < 2 * dim) {
                            totAlpha[3] = 0;

                        } else if (numEdges >= 2 * dim) {
                            for (PetscInt iEdges = 0; iEdges < numEdges; ++iEdges) {
                                // get faces attached to each vertex
                                const PetscInt* attachedFace;
                                PetscInt numAttachedFace;

                                DMPlexGetSupport(dm, edges[iEdges], &attachedFace);
                                DMPlexGetSupportSize(dm, edges[iEdges], &numAttachedFace);

                                // Get cells attached to each face
                                for (PetscInt iFace = 0; iFace < numAttachedFace; ++iFace) {
                                    const PetscInt* cell;
                                    PetscInt numCell;

                                    DMPlexGetSupport(dm, attachedFace[iFace], &cell);
                                    DMPlexGetSupportSize(dm, attachedFace[iFace], &numCell);
                                    if (numCell != 2) continue;

                                    PetscReal vertexAlpha[3] = {0, 0, 0};
                                    PetscReal distance[3] = {0, 0, 0};

                                    // march over attached-cells of each face to calculate normal
                                    for (PetscInt iCell = 0; iCell < numCell; ++iCell) {
                                        PetscReal verNormal[3] = {0, 0, 0};
                                        // get centroid for attached-cells to each face

                                        PetscFVCellGeom* cg;
                                        DMPlexPointLocalRead(dmCell, cell[iCell], cellGeomArray, &cg);
                                        // get current alpha of each attached face
                                        PetscScalar* alpha = nullptr;
                                        DMPlexPointLocalFieldRead(dm, cell[iCell], VFfield->id, solArray, &alpha);

                                        for (PetscInt d = 0; d < dim; ++d) {
                                            if (cg->centroid[d] > xyz[d]) {
                                                verNormal[d] = *alpha;

                                            } else if (cg->centroid[d] < xyz[d]) {
                                                verNormal[d] = -*alpha;

                                            } else {
                                                verNormal[d] = 0;
                                            }
                                            distance[d] += abs(cg->centroid[d] - xyz[d] + TINY);

                                            vertexAlpha[d] += verNormal[d];
                                        }
                                    }

                                    for (PetscInt d = 0; d < dim; ++d) {
                                        distCell[d] += distance[d];
                                        totAlpha[d] += vertexAlpha[d];
                                    }
                                }
                            }
                        }
                        // Now calculate normal at each cell center using normals of attached vortices
                        for (PetscInt d = 0; d < dim; ++d) {
                            // divide by "dim" since each cell is used "dim" times
                            normal[d] = totAlpha[d] / (distCell[d] + TINY) / dim;

                            if (fcg->centroid[d] < xyz[d] || fcg->centroid[d] > xyz[d]) {
                                divergentNormal[d] = normal[d] / (xyz[d] - fcg->centroid[d]);

                            } else {
                                divergentNormal[d] = 0;
                            }
                            totalDivNormal += divergentNormal[d] / numVortices;
                            centerNormal[d] += normal[d] / numVortices;

                            // calculate magnitude of normal at vortices
                            const PetscReal magVertexNormal = utilities::MathUtilities::MagVector(dim, normal);
                            // calculate the derivative of the magnitude of normal at cell center
                            if (fcg->centroid[d] < xyz[d] || fcg->centroid[d] > xyz[d]) {
                                grad[d] = magVertexNormal / (xyz[d] - fcg->centroid[d]);

                            } else {
                                grad[d] = 0;
                            }
                            gradNormal[d] += grad[d] / numVortices;
                        }
                    }
                    VecRestoreArray(localCoordsVector, &coordsArray) >> utilities::PetscUtilities::checkError;
                    for (PetscInt d = 0; d < dim; ++d) {
                        // divide by "dim" since  each vertex is used "dim" times
                        totGradNormal[d] += gradNormal[d] / dim;
                        cellCenterNormal[d] += centerNormal[d] / dim;
                        centerDivNormal += totalDivNormal / dim;
                    }
                }
            }
        }
        // magnitude of normal at the center
        const PetscReal magCellNormal = utilities::MathUtilities::MagVector(dim, cellCenterNormal);
        if (magCellNormal > 0) {
            for (PetscInt d = 0; d < dim; ++d) {
                // (ni,j/ |ni,j| . Delta)
                gradMagNormal[d] = (cellCenterNormal[d] / (magCellNormal)) * totGradNormal[d];
                totalGradMagNormal += gradMagNormal[d];
            }
            // calculate curvature -->  kappa = 1/n [(n/|n|. Delta) |n| - (Delta.n)]
            curvature = (totalGradMagNormal * magCellNormal - centerDivNormal) / (magCellNormal);
        } else {
            curvature = 0;
        }
        for (PetscInt d = 0; d < dim; ++d) {
            // calculate surface force and energy
            surfaceForce[d] = process->sigma * curvature * cellCenterNormal[d];
            surfaceEnergy += surfaceForce[d] * vel[d];

            // add in the contributions
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = surfaceForce[d];
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] = surfaceEnergy;
        }
    }
    // cleanup
    VecRestoreArray(locFVec, &fArray);
    solver.RestoreRange(cellRange);
    PetscFunctionReturn(0);
}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::SurfaceForce, "calculates surface tension force and adds source terms",
         ARG(PetscReal, "value", "sigma, surface tension coefficient"));