#include "der.hpp"

using namespace ablate::levelSet;


// Evaluate a derivative from a stencil list
// f - Vector cotaining the data
// n - The length of the stencil
// lst - List of cell IDs
// der - The particular derivative to evaluate
// nDer - The total number of derivatives computed
// wt - The weights, arranged in row-major ordering.
PetscReal EvalDer(Vec f, PetscInt n, PetscInt lst[], PetscInt der, PetscInt nDer, PetscReal wt[]) {

  PetscReal       val = 0.0, *array;
  PetscInt        i;

  VecGetArray(f, &array) >> ablate::checkError;

  for (i = 0; i < n; ++i) {
    val += wt[i*nDer + der]*array[lst[i]];
  }

  VecRestoreArray(f, &array) >> ablate::checkError;

  return val;

}


// Returns the requested derivative stencils
// rbf - The Radial Basis Function to use.
// nDer - Number of derivatives to compute
// dx, dy, dz - Lists of length nDer indicating the derivatives
// nStencilOut - Length of each derivative stencil
// stencilListOut - Cell IDs of the stencil list
// stencilWeightsOut - Cell weights
//
// Example of use: nDer = 5, dx[] = {1, 0, 2, 0, 1}, dy[] = {0, 1, 0, 2, 1}, dz[] = {0, 0, 0, 0, 0} will return the dx, dy, dxx, dyy, dxy stencils lists
PetscErrorCode GetDerivativeStencils(std::shared_ptr<RBF> rbf, PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[], PetscInt **nStencilOut, PetscInt ***stencilListOut, PetscReal ***stencilWeightsOut) {
  PetscInt        c, cStart, cEnd, n, i;
  PetscInt        *nStencil, **stencilList;
  PetscReal       **stencilWeights;
  PetscReal       h;
  DM              dm = rbf->GetDM();
  PetscBool       useVertices = PETSC_TRUE;

  PetscFunctionBegin;

  DMPlexGetMinRadius(dm, &h) >> ablate::checkError; // This returns the minimum distance from any cell centroid to a face.
  h *= 2.0;                                         // Double it to get the grid spacing.

  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> ablate::checkError;      // Range of cells
  n = cEnd - cStart;

  PetscMalloc3(n, &nStencil, n, &stencilList, n, &stencilWeights) >> ablate::checkError;

  for (i = 0; i < cEnd-cStart; ++i) {
    nStencil[i] = 0;
    stencilList[i] = NULL;
    stencilWeights[i] = NULL;
  }

  for (c = cStart; c < cEnd; ++c) {
    i = c - cStart;
    DMPlexGetNeighborCells(dm, c, 3, 4.0*h, useVertices, &nStencil[i], &stencilList[i]);
    rbf->Weights(c, nStencil[i], stencilList[i], nDer, dx, dy, dz, &stencilWeights[i]);
  }

  *nStencilOut = nStencil;
  *stencilListOut = stencilList;
  *stencilWeightsOut = stencilWeights;

  PetscFunctionReturn(0);

}
