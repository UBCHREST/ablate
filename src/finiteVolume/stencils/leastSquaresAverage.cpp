#include "leastSquaresAverage.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscUtilities.hpp"

void ablate::finiteVolume::stencil::LeastSquaresAverage::Generate(PetscInt face, ablate::finiteVolume::stencil::Stencil& stencil, const domain::SubDomain& subDomain,
                                                                  const std::shared_ptr<domain::Region>& solverRegion, DM cellDM, const PetscScalar* cellGeomArray, DM faceDM,
                                                                  const PetscScalar* faceGeomArray) {
    auto dm = subDomain.GetDM();
    auto dim = subDomain.GetDimensions();

    // compute the grad calculator if needed
    if (!gradientCalculator) {
        PetscFVCreate(PETSC_COMM_SELF, &gradientCalculator) >> utilities::PetscUtilities::checkError;
        // Set least squares as the default type
        PetscFVSetType(gradientCalculator, PETSCFVLEASTSQUARES) >> utilities::PetscUtilities::checkError;
        // Set any other required options
        PetscFVSetFromOptions(gradientCalculator) >> utilities::PetscUtilities::checkError;
        PetscFVSetNumComponents(gradientCalculator, 1) >> utilities::PetscUtilities::checkError;
        PetscFVSetSpatialDimension(gradientCalculator, dim) >> utilities::PetscUtilities::checkError;
    }

    PetscInt numCells;
    DMPlexGetSupportSize(dm, face, &numCells);
    const PetscInt* cells;
    DMPlexGetSupport(dm, face, &cells);

    // Create the stencil for the left and right cells first
    Stencil leftStencil;
    ComputeNeighborCellStencil(cells[0], leftStencil, subDomain, solverRegion, cellDM, cellGeomArray, faceDM, faceGeomArray);

    Stencil rightStencil;
    if (numCells > 1) ComputeNeighborCellStencil(cells[1], rightStencil, subDomain, solverRegion, cellDM, cellGeomArray, faceDM, faceGeomArray);

    // Merge the stencils together
    stencil.stencil = leftStencil.stencil;
    stencil.stencil.insert(stencil.stencil.end(), rightStencil.stencil.begin(), rightStencil.stencil.end());
    std::sort(stencil.stencil.begin(), stencil.stencil.end());
    stencil.stencil.erase(std::unique(stencil.stencil.begin(), stencil.stencil.end()), stencil.stencil.end());
    stencil.stencilSize = (PetscInt)stencil.stencil.size();

    // resize the weight arrays and init to zero
    stencil.weights.resize(stencil.stencilSize, 0.0);
    stencil.gradientWeights.resize(stencil.stencilSize * dim, 0.0);

    // add the contributions for each cell
    PetscInt numberStencils = (leftStencil.stencilSize ? 1 : 0) + (rightStencil.stencilSize ? 1 : 0);

    for (std::size_t i = 0; i < leftStencil.stencil.size(); i++) {
        auto stencilIt = std::find(stencil.stencil.begin(), stencil.stencil.end(), leftStencil.stencil[i]);
        auto stencilLocation = std::distance(stencil.stencil.begin(), stencilIt);

        stencil.weights[stencilLocation] += leftStencil.weights[i] / numberStencils;
        for (PetscInt d = 0; d < dim; d++) {
            stencil.gradientWeights[stencilLocation * dim + d] += leftStencil.gradientWeights[i * dim + d] / numberStencils;
        }
    }

    for (std::size_t i = 0; i < rightStencil.stencil.size(); i++) {
        auto stencilIt = std::find(stencil.stencil.begin(), stencil.stencil.end(), rightStencil.stencil[i]);
        auto stencilLocation = std::distance(stencil.stencil.begin(), stencilIt);

        stencil.weights[stencilLocation] += rightStencil.weights[i] / numberStencils;
        for (PetscInt d = 0; d < dim; d++) {
            stencil.gradientWeights[stencilLocation * dim + d] += rightStencil.gradientWeights[i * dim + d] / numberStencils;
        }
    }
}

void ablate::finiteVolume::stencil::LeastSquaresAverage::ComputeNeighborCellStencil(PetscInt cell, ablate::finiteVolume::stencil::Stencil& stencil, const ablate::domain::SubDomain& subDomain,
                                                                                    const std::shared_ptr<domain::Region>& solverRegion, DM cellDM, const PetscScalar* cellGeomArray, DM faceDM,
                                                                                    const PetscScalar* faceGeomArray) {
    // only add the cells if they are in the ds
    if (!subDomain.InRegion(cell)) {
        return;
    }
    const auto dm = subDomain.GetDM();
    const auto dim = subDomain.GetDimensions();

    stencil.stencil.push_back(cell);
    // get the face info for this cell
    PetscInt numberCellFaces;
    const PetscInt* cellFaces;

    DMPlexGetConeSize(dm, cell, &numberCellFaces) >> utilities::PetscUtilities::checkError;
    DMPlexGetCone(dm, cell, &cellFaces) >> utilities::PetscUtilities::checkError;

    PetscFVCellGeom* cellGeom;
    DMPlexPointLocalRead(cellDM, cell, cellGeomArray, &cellGeom) >> utilities::PetscUtilities::checkError;

    // March over each face connected to this cell
    for (PetscInt f = 0; f < numberCellFaces; f++) {
        // determine if there are two cells connected
        PetscInt numFaceCells;
        const PetscInt* neighborCells;
        DMPlexGetSupportSize(dm, cellFaces[f], &numFaceCells) >> utilities::PetscUtilities::checkError;
        if (numFaceCells != 2) {
            continue;
        }
        DMPlexGetSupport(dm, cellFaces[f], &neighborCells) >> utilities::PetscUtilities::checkError;
        // determine which one is the neighbor
        PetscInt neighborCell = neighborCells[0] == cell ? neighborCells[1] : neighborCells[0];

        if (subDomain.InRegion(neighborCell) && domain::Region::InRegion(solverRegion, dm, cellFaces[f])) {
            stencil.stencil.push_back(neighborCell);
        }
    }

    // compute the stencil
    // ignore cell if there are no stencils
    stencil.stencilSize = (PetscInt)stencil.stencil.size();
    if (stencil.stencilSize) {
        // for now, set the interpolant weights to be the average of the two faces
        stencil.weights.resize(stencil.stencilSize, 0.0);
        stencil.gradientWeights.resize(stencil.stencilSize * subDomain.GetDimensions(), 0.0);
        if (stencil.stencilSize * dim > (PetscInt)dx.size()) {
            dx.resize(stencil.stencilSize * dim);
        }

        // set the stencil weight for this cell to unity
        stencil.weights[0] = 1.0;

        // Compute gradients
        if (stencil.stencilSize > maxFaces) {
            maxFaces = stencil.stencilSize;
            PetscFVLeastSquaresSetMaxFaces(gradientCalculator, maxFaces) >> utilities::PetscUtilities::checkError;
        }

        // compute the distance between the cell centers and the face
        for (PetscInt n = 0; n < stencil.stencilSize - 1; n++) {
            PetscFVCellGeom* neighborGeom;
            DMPlexPointLocalRead(cellDM, stencil.stencil[n + 1], cellGeomArray, &neighborGeom);

            for (PetscInt d = 0; d < dim; ++d) {
                dx[n * dim + d] = neighborGeom->centroid[d] - cellGeom->centroid[d];
            }
        }

        // compute the gradient weights and offset by the dim for the first weight
        if (!PetscFVComputeGradient(gradientCalculator, stencil.stencilSize - 1, dx.data(), stencil.gradientWeights.data() + dim)) {
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
}

ablate::finiteVolume::stencil::LeastSquaresAverage::~LeastSquaresAverage() {
    if (gradientCalculator) {
        PetscFVDestroy(&gradientCalculator);
    }
}
