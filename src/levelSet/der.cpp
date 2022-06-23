#include "der.hpp"

using namespace ablate::levelSet;

// Returns the requested derivative stencils
// rbf - The Radial Basis Function to use.
// nDer - Number of derivatives to compute
// dx, dy, dz - Lists of length nDer indicating the derivatives
// nStencilOut - Length of each derivative stencil
// stencilListOut - Cell IDs of the stencil list
// stencilWeightsOut - Cell weights
//
// Example of use: nDer = 5, dx[] = {1, 0, 2, 0, 1}, dy[] = {0, 1, 0, 2, 1}, dz[] = {0, 0, 0, 0, 0} will return the dx, dy, dxx, dyy, dxy stencils lists
void DerCalculator::SetupDerivativeStencils(std::shared_ptr<RBF> rbf, PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[], PetscInt **nStencilOut, PetscInt ***stencilListOut, PetscReal ***stencilWeightsOut) {
//  PetscInt        c, cStart, cEnd, n, i;
//  PetscInt        *nStencil, **stencilList;
//  PetscReal       **stencilWeights;
//  DM              dm = rbf->GetDM();
//  PetscBool       useVertices = PETSC_TRUE;
//  PetscInt        minNumberCells = (PetscInt)(1.75*rbf->GetNPoly());

//  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> ablate::checkError;      // Range of cells
//  n = cEnd - cStart;

//  PetscMalloc3(n, &nStencil, n, &stencilList, n, &stencilWeights) >> ablate::checkError;

//  for (i = 0; i < cEnd-cStart; ++i) {
//    nStencil[i] = 0;
//    stencilList[i] = NULL;
//    stencilWeights[i] = NULL;
//  }

//  for (c = cStart; c < cEnd; ++c) {
//    i = c - cStart;
//    DMPlexGetNeighborCells(dm, c, -1, -1.0, minNumberCells, useVertices, &nStencil[i], &stencilList[i]);
//    rbf->Weights(c, nStencil[i], stencilList[i], nDer, dx, dy, dz, &stencilWeights[i]);
//  }
//  *nStencilOut = nStencil;
//  *stencilListOut = stencilList;
//  *stencilWeightsOut = stencilWeights;

}


DerCalculator::DerCalculator(std::shared_ptr<RBF> rbf, PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[]) {

//  // Set which derivatives are being computed
//  DerCalculator::nDer = nDer;
//  PetscMalloc1(nDer, &(DerCalculator::dxyz));

//  for (int i = 0; i < nDer; ++i) {
//    DerCalculator::dxyz[i] = dx[i]*100 + dy[i]*10 + dz[i];
//  }

//  DerCalculator::dm = rbf->GetDM();


//  // Now determine the actual stencils
//  DerCalculator::SetupDerivativeStencils(rbf, nDer, dx, dy, dz, &(DerCalculator::nStencil), &(DerCalculator::stencilList), &(DerCalculator::stencilWeights));

}

DerCalculator::~DerCalculator() {



//  PetscInt c, cStart, cEnd, i;

//  DMPlexGetHeightStratum(DerCalculator::dm, 0, &cStart, &cEnd) >> ablate::checkError;      // Range of cells


//  for (c = cStart; c < cEnd; ++c) {
//    i = c - cStart;
//    PetscFree(DerCalculator::stencilList[i]) >> ablate::checkError;
//    PetscFree(DerCalculator::stencilWeights[i]) >> ablate::checkError;
//  }

//  PetscFree3(DerCalculator::nStencil, DerCalculator::stencilList, DerCalculator::stencilWeights) >> ablate::checkError;
//  PetscFree(DerCalculator::dxyz) >> ablate::checkError;


}




// Evaluate a derivative from a stencil list
// f - Vector cotaining the data
// der - The particular derivative to evaluate
// nDer - The total number of derivatives computed
// nStencil - The length of the stencil
// lst - List of cell IDs
// wt - The weights, arranged in row-major ordering.
PetscReal DerCalculator::EvalDer_Internal(Vec f, PetscInt der, PetscInt nDer, PetscInt nStencil, PetscInt lst[], PetscReal wt[]) {

  PetscReal       val = 0.0, *array;
  PetscInt        i;

  VecGetArray(f, &array) >> ablate::checkError;

  for (i = 0; i < nStencil; ++i) {
    val += wt[i*nDer + der]*array[lst[i]];
//    printf("%d\t%f\n", lst[i], wt[i*nDer+der]);
  }

  VecRestoreArray(f, &array) >> ablate::checkError;

  return val;

}

// Return the requested derivative
// f - Vector containing the data
// c - Location to evaluate at
// dx, dy, dz - The derivatives
PetscReal DerCalculator::EvalDer(Vec f, PetscInt c, PetscInt dx, PetscInt dy, PetscInt dz){

  PetscInt  id = -1, target = dx*100 + dy*10 + dz;
  PetscInt  nDer = DerCalculator::nDer;
  PetscInt  *dxyz = DerCalculator::dxyz;
  PetscReal val = 0.0;

  // Search for the particular index. Probably need to do something different in the future to avoid re-doing the same calculation many times
  while (dxyz[++id] != target && id<nDer){ }

  if (id==nDer) {
    throw std::invalid_argument("Derivative of (" + std::to_string(dx) + ", " + std::to_string(dy) + ", " + std::to_string(dz) + " is not setup.");
  }




  val = DerCalculator::EvalDer_Internal(f, id, nDer, DerCalculator::nStencil[c], DerCalculator::stencilList[c], DerCalculator::stencilWeights[c]);

  return val;

}
