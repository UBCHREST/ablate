#include "leastSquares.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscError.hpp"

void ablate::finiteVolume::stencil::LeastSquares::Generate(PetscInt face, ablate::finiteVolume::stencil::Stencil& stencil, const domain::SubDomain& subDomain,
                                                           const std::shared_ptr<domain::Region> solverRegion, DM cellDM, const PetscScalar* cellGeomArray, DM faceDM,
                                                           const PetscScalar* faceGeomArray) {
    auto dm = subDomain.GetDM();
    auto dim = subDomain.GetDimensions();

    // compute the grad calculator if needed
    if (!gradientCalculator) {
        PetscFVCreate(PETSC_COMM_SELF, &gradientCalculator) >> checkError;
        // Set least squares as the default type
        PetscFVSetType(gradientCalculator, PETSCFVLEASTSQUARES) >> checkError;
        // Set any other required options
        PetscFVSetFromOptions(gradientCalculator) >> checkError;
        PetscFVSetNumComponents(gradientCalculator, 1) >> checkError;
        PetscFVSetSpatialDimension(gradientCalculator, dim) >> checkError;
    }
    // get the face geom
    PetscFVFaceGeom* fg;
    DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg);

    // Get all nodes in this face
    PetscInt numberNodes;
    PetscInt* faceNodes = nullptr;
    DMPlexGetTransitiveClosure(dm, face, PETSC_TRUE, &numberNodes, &faceNodes) >> checkError;

    // For each node get the cells that connect to it
    for (PetscInt n = 0; n < numberNodes; n++) {
        PetscInt numberCells;
        PetscInt* nodeCells = nullptr;
        DMPlexGetTransitiveClosure(dm, faceNodes[n * 2], PETSC_FALSE, &numberCells, &nodeCells) >> checkError;

        for (PetscInt c = 0; c < numberCells; c++) {
            PetscInt cell = nodeCells[c * 2];
            // Make sure that cell is in this region and is a cell
            PetscInt cellHeight;
            DMPlexGetPointHeight(dm, cell, &cellHeight) >> checkError;
            if (cellHeight != 0) {
                continue;
            }
            if (!subDomain.InRegion(cell)) {
                continue;
            }

            stencil.stencil.push_back(cell);
        }

        // cleanup
        DMPlexRestoreTransitiveClosure(dm, faceNodes[n * 2], PETSC_FALSE, &numberCells, &nodeCells) >> checkError;
    }

    // cleanup
    DMPlexRestoreTransitiveClosure(dm, face, PETSC_TRUE, &numberNodes, &faceNodes) >> checkError;

    // Clean up the stencil to remove duplicates
    std::sort(stencil.stencil.begin(), stencil.stencil.end());
    stencil.stencil.erase(std::unique(stencil.stencil.begin(), stencil.stencil.end()), stencil.stencil.end());

    // ignore cell if there are no stencils
    stencil.stencilSize = (PetscInt)stencil.stencil.size();
    if (stencil.stencilSize) {
        // for now, set the interpolant weights to be the average of the two faces
        stencil.weights.resize(stencil.stencilSize, 0.0);
        stencil.gradientWeights.resize(stencil.stencilSize * dim, 0.0);
        if (stencil.stencilSize * dim > (PetscInt)dx.size()) {
            dx.resize(stencil.stencilSize * dim);
        }

        // Get the support for this face
        PetscInt numberNeighborCells;
        const PetscInt* neighborCells;
        DMPlexGetSupportSize(dm, face, &numberNeighborCells) >> ablate::checkError;
        DMPlexGetSupport(dm, face, &neighborCells) >> ablate::checkError;
        // Set the stencilWeight
        PetscReal sum = 0.0;
        for (PetscInt c = 0; c < numberNeighborCells; c++) {
            for (PetscInt i = 0; i < stencil.stencilSize; i++) {
                if (neighborCells[c] == stencil.stencil[i]) {
                    stencil.weights[i] = 1.0;
                    sum += 1.0;
                }
            }
        }

        ablate::utilities::MathUtilities::ScaleVector(stencil.stencilSize, stencil.weights.data(), 1.0 / sum);

        // Compute gradients
        if (stencil.stencilSize > maxFaces) {
            maxFaces = stencil.stencilSize;
            PetscFVLeastSquaresSetMaxFaces(gradientCalculator, maxFaces) >> checkError;
        }

        // compute the distance between the cell centers and the face
        for (PetscInt n = 0; n < stencil.stencilSize; n++) {
            PetscFVCellGeom* cg;
            DMPlexPointLocalRead(cellDM, stencil.stencil[n], cellGeomArray, &cg);

            for (PetscInt d = 0; d < dim; ++d) {
                dx[n * dim + d] = cg->centroid[d] - fg->centroid[d];
            }
        }

        PetscFVComputeGradient(gradientCalculator, stencil.stencilSize, dx.data(), stencil.gradientWeights.data()) >> checkError;

        // now combine the gradient weights with the stencil weights so that we don't need to precompute the stencil value and compute dx
        const auto gradientWeightsOrg = stencil.gradientWeights;

        // march over the gradient stencil
        for (PetscInt gn = 0; gn < stencil.stencilSize; gn++) {
            // march over each face stencil
            for (PetscInt fn = 0; fn < stencil.stencilSize; fn++) {
                // add the contribution for the gradient stencil * (-) the face stencil to the face stencil node location
                for (PetscInt d = 0; d < dim; ++d) {
                    stencil.gradientWeights[fn * dim + d] -= gradientWeightsOrg[gn * dim + d] * stencil.weights[fn];
                }
            }
        }
    }
}

ablate::finiteVolume::stencil::LeastSquares::~LeastSquares() {
    if (gradientCalculator) {
        PetscFVDestroy(&gradientCalculator);
    }
}
