#include "rbf.hpp"
#include <petsc/private/dmpleximpl.h>

using namespace ablate::domain::rbf;


// Distance between two points
PetscReal RBF::DistanceSquared(PetscReal x[], PetscReal y[]){
  PetscInt dim = subDomain->GetDimensions();
  PetscReal r = 0.0;
  for (PetscInt d = 0; d < dim; ++d) {
    r += (x[d] - y[d]) * (x[d] - y[d]);
  }

  return r;
}

// Distance between point and the origin
PetscReal RBF::DistanceSquared(PetscReal x[]){
  PetscInt dim = subDomain->GetDimensions();
  PetscReal r = 0.0;
  for (PetscInt d = 0; d < dim; ++d) {
    r += x[d] * x[d];
  }
  return r;
}



static PetscInt fac[11] =  {1,1,2,6,24,120,720,5040,40320,362880,3628800}; // Pre-computed factorials

// Compute the LU-factorization of the augmented RBF matrix
// c - Location in cellRange
// xCenters - Shifted centers of all the cells (output)
// A - The LU-factorization of the augmented RBF matrix (output)
void RBF::Matrix(const PetscInt c, PetscReal **xCenters, Mat *LUA) {

  PetscInt              i, j, d, px, py, pz, matSize;
  PetscInt              nCells, *list;
  const PetscInt        dim = subDomain->GetDimensions();
  const PetscInt        nPoly = RBF::nPoly, p = RBF::polyOrder, p1 = p + 1;
  Mat                   A;
  PetscReal             *x, *vals;
  PetscReal             x0[dim];        // Center of the cell of interest
  PetscReal             *xp; // Powers of the cell centers
  const DM              dm = subDomain->GetSubDM();

  // Get the list of neighbor cells
  DMPlexGetNeighborCells(dm, c, -1, -1.0, RBF::minNumberCells, RBF::useVertices, &nCells, &list);
  RBF::nStencil[c] = nCells;
  RBF::stencilList[c] = list;

  PetscMalloc1(nCells*dim*p1, &xp) >> ablate::checkError;


  if(nPoly>=nCells){
    throw std::invalid_argument("Number of surrounding cells, " + std::to_string(nCells) + ", can not support a requested polynomial order of " + std::to_string(p) + " which requires " + std::to_string(nPoly) + " number of cells.");
  }

  // Get the cell center
  DMPlexComputeCellGeometryFVM(dm, c, NULL, x0, NULL) >> ablate::checkError;

  // Shifted cell-centers of neighbor cells
  PetscMalloc1(nCells*dim, &x);
  for (i = 0; i < nCells; ++i) {
    DMPlexComputeCellGeometryFVM(dm, list[i], NULL, &x[i*dim], NULL) >> ablate::checkError;
    for (d = 0; d < dim; ++d) {
      x[i*dim+d] -= x0[d];
      // Precompute the powers for later use
      xp[(i*dim+d)*p1 + 0] = 1.0;
      for (px = 1; px < p+1; ++px) {
        xp[(i*dim+d)*p1 + px] = xp[(i*dim+d)*p1 + (px-1)]*x[i*dim+d];
      }
    }
  }

  matSize = nCells + nPoly;

  // Create the matrix
  MatCreateSeqDense(PETSC_COMM_SELF, matSize, matSize, NULL, &A) >> ablate::checkError;
  PetscObjectSetName((PetscObject)A,"ablate::domain::RBF::A") >> ablate::checkError;
  MatZeroEntries(A) >> ablate::checkError;
  MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE) >> ablate::checkError;

  MatDenseGetArrayWrite(A, &vals) >> ablate::checkError;

  // RBF contributions to the matrix
  for (i = 0; i < nCells; ++i) {
    for (j = i; j < nCells; ++j) {
      vals[i*matSize + j] = vals[j*matSize + i] = RBFVal(&x[i*dim], &x[j*dim]);
    }
  }

  // Augmented polynomial contributions
  if (dim == 2) {
    for (i = 0; i < nCells; ++i) {
      j = nCells;
      for (py = 0; py < p1; ++py) {
        for (px = 0; px < p1-py; ++px) {
          vals[i*matSize + j] = vals[j*matSize + i] = xp[(i*dim+0)*p1 + px] * xp[(i*dim+1)*p1 + py];
          ++j;
        }
      }
    }
  } else {
    for (i = 0; i < nCells; ++i) {
      j = nCells;
      for (pz = 0; pz < p1; ++pz) {
        for (py = 0; py < p1-pz; ++py) {
          for (px = 0; px < p1-py-pz; ++px ){
            vals[i*matSize + j] = vals[j*matSize + i] = xp[(i*dim+0)*p1 + px] * xp[(i*dim+1)*p1 + py] * xp[(i*dim+2)*p1 + pz];
            ++j;
          }
        }
      }
    }
  }
  MatDenseRestoreArrayWrite(A, &vals) >> ablate::checkError;
  MatViewFromOptions(A,NULL,"-ablate::domain::rbf::RBF::A_view") >> ablate::checkError;

  // Factor the matrix
  MatLUFactor(A, NULL, NULL, NULL) >> ablate::checkError;

  PetscFree(xp) >> ablate::checkError;

  // Assign output
  *xCenters = x;
  *LUA = A;

}


/************ Begin Derivative Code **********************/


// Set the derivatives to use
// nDer - Number of derivatives to set
// dx, dy, dz - Lists of length nDer indicating the derivatives
void RBF::SetDerivatives(PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[], PetscBool useVertices){

  if (nDer > 0) {

    RBF::useVertices = useVertices;
    RBF::nDer = nDer;

    PetscMalloc1(3*nDer, &(RBF::dxyz)) >> ablate::checkError;

    // Store the derivatives
    for (PetscInt i = 0; i < nDer; ++i) {
      RBF::dxyz[i*3 + 0] = dx[i];
      RBF::dxyz[i*3 + 1] = dy[i];
      RBF::dxyz[i*3 + 2] = dz[i];
    }

  }


}

// Set derivatives, defaulting to using vertices
void RBF::SetDerivatives(PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[]){
  RBF::SetDerivatives(nDer, dx, dy, dz, PETSC_TRUE);
}




// Compute the RBF weights at the cell center of p using a cell-list
// c - The center cell in cellRange ordering
void RBF::SetupDerivativeStencils(PetscInt c) {

  Mat       A = RBF::RBFMatrix[c], B = nullptr;
  PetscReal *x = nullptr;   // Shifted cell-centers of the neigbor list
  // This also computes nStencil[c]
  if (A == nullptr) {
    RBF::Matrix(c, &x, &A);
  }

  PetscInt    dim = subDomain->GetDimensions();
  PetscInt    nCells = RBF::nStencil[c];
  PetscInt    matSize = nCells + RBF::nPoly;
  PetscInt    nDer = RBF::nDer;
  PetscInt    *dxyz = RBF::dxyz;
  PetscInt    i, j;
  PetscInt    px, py, pz, p1 = RBF::polyOrder+1;
  PetscScalar *vals = nullptr;

  //Create the RHS
  MatCreateSeqDense(PETSC_COMM_SELF, matSize, nDer, NULL, &B) >> ablate::checkError;
  PetscObjectSetName((PetscObject)B,"ablate::domain::rbf::RBF::rhs") >> ablate::checkError;
  MatZeroEntries(B) >> ablate::checkError;
  MatDenseGetArrayWrite(B, &vals) >> ablate::checkError;

  // Derivatives of the RBF
  for (i = 0; i < nCells; ++i) {
    for (j = 0; j < nDer; ++j) {
      vals[i + j*matSize] = RBFDer(&x[i*dim], dxyz[j*3 + 0], dxyz[j*3 + 1], dxyz[j*3 + 2]);
    }
  }

  // Derivatives of the augmented polynomials
  if (dim == 2) {
    for (j = 0; j < nDer; ++j) {
      i = nCells;
      for (py = 0; py < p1; ++py) {
        for (px = 0; px < p1-py; ++px ){
          if(dxyz[j*3 + 0] == px && dxyz[j*3 + 1] == py) {
            vals[i + j*matSize] = fac[px]*fac[py];
          }
          ++i;
        }
      }

    }
  } else {
    for (j = 0; j < nDer; ++j) {
      i = nCells;
      for (pz = 0; pz < p1; ++pz) {
        for (py = 0; py < p1-pz; ++py) {
          for (px = 0; px < p1-py-pz; ++px ){
            if(dxyz[j*3 + 0] == px && dxyz[j*3 + 1] == py && dxyz[j*3 + 2] == pz) {
              vals[i + j*matSize] = fac[px]*fac[py]*fac[pz];
            }
            ++i;
          }
        }
      }
    }
  }

  MatDenseRestoreArrayWrite(B, &vals) >> ablate::checkError;

  MatViewFromOptions(B,NULL,"-ablate::domain::rbf::RBF::rhs_view") >> ablate::checkError;

  MatMatSolve(A, B, B) >> ablate::checkError;

  MatViewFromOptions(B,NULL,"-ablate::domain::rbf::RBF::sol_view") >> ablate::checkError;

  // Now populate the output
  PetscMalloc1(nDer*nCells, &(RBF::stencilWeights[c])) >> ablate::checkError;
  PetscReal *wt = RBF::stencilWeights[c];
  MatDenseGetArrayWrite(B, &vals) >> ablate::checkError;
  for (i = 0; i < nCells; ++i) {
    for (j = 0; j < nDer; ++j) {
      wt[i*nDer + j] = vals[i + j*matSize];
    }
  }
  MatDenseGetArrayWrite(B, &vals) >> ablate::checkError;

  // The matrix is only needed in the future if interpolation is required. If not then
  //    the stencil list is enough.
  if (RBF::hasInterpolation) {
    RBF::RBFMatrix[c] = A;
    RBF::stencilXLocs[c] = x;
  }
  else {
    MatDestroy(&A) >> ablate::checkError;
    PetscFree(x) >> ablate::checkError;
  }
  MatDestroy(&B) >> ablate::checkError;

}



void RBF::SetupDerivativeStencils() {
  const PetscInt cStart = RBF::cStart, cEnd = RBF::cEnd;

  for (PetscInt c = cStart; c < cEnd; ++c) {
    RBF::SetupDerivativeStencils(RBF::cellList[c]);
  }

}



// Return the requested derivative
// field - The field to take the derivative of
// c - The location in ablate::solver::Range
// dx, dy, dz - The derivatives
PetscReal RBF::EvalDer(const ablate::domain::Field *field, PetscInt c, PetscInt dx, PetscInt dy, PetscInt dz){

  PetscInt  derID = -1, nDer = RBF::nDer, *dxyz = RBF::dxyz;
  // Search for the particular index. Probably need to do something different in the future to avoid re-doing the same calculation many times
  while ((dxyz[++derID*3 + 0] != dx || dxyz[derID*3 + 1] != dy || dxyz[derID*3 + 2] != dz) && derID<nDer){ }

  if (derID==nDer) {
    throw std::invalid_argument("Derivative of (" + std::to_string(dx) + ", " + std::to_string(dy) + ", " + std::to_string(dz) + ") is not setup.");
  }

  // If the stencil hasn't been setup yet do so
  if (RBF::nStencil[c] < 1) {
    RBF::SetupDerivativeStencils(c);
  }

  PetscReal         *wt = RBF::stencilWeights[c];
  PetscInt          nStencil = RBF::nStencil[c], *lst = RBF::stencilList[c];
  Vec               vec = subDomain->GetVec(*field);
  DM                dm  = subDomain->GetFieldDM(*field);
  PetscScalar       val = 0.0, *f;
  const PetscScalar *array;
  VecGetArrayRead(vec, &array) >> checkError;

  for (PetscInt i = 0; i < nStencil; ++i) {
    // DMPlexPointLocalFieldRead isn't behaving like I would expect. If I don't make f a pointer then it just returns zero.
    //    Additionally, it looks like it allows for the editing of the value.
    DMPlexPointLocalFieldRead(dm, lst[i], field->id, array, &f) >> checkError;
    val += wt[i*nDer + derID]*(*f);
  }

  VecRestoreArrayRead(vec, &array) >> checkError;

  return val;
}



/************ End Derivative Code **********************/




/************ Begin Interpolation Code **********************/


// Return the interpolation of a field at a given location
// field - The field to interpolate
// xEval - The location to interpolate at
PetscReal RBF::Interpolate(const ablate::domain::Field *field, PetscReal xEval[3]) {

  Mat         A;
  Vec         weights, rhs;
  PetscInt    i, c, nCells, *lst;
  PetscScalar *vals;
  const PetscScalar *fvals;
  PetscReal   *x, x0[3];
  Vec         f = subDomain->GetVec(*field);
  DM          dm  = subDomain->GetFieldDM(*field);

  DMPlexGetContainingCell(dm, xEval, &c) >> ablate::checkError;

  A = RBF::RBFMatrix[c];
  x = RBF::stencilXLocs[c];

  if (A == nullptr) {
    RBF::Matrix(c, &x, &A);
    RBF::RBFMatrix[c] = A;
    RBF::stencilXLocs[c] = x;
  }

  nCells = RBF::nStencil[c];
  lst = RBF::stencilList[c];

  MatCreateVecs(A, &weights, &rhs) >> ablate::checkError;
  VecZeroEntries(weights) >> ablate::checkError;
  VecZeroEntries(rhs) >> ablate::checkError;

  // The function values
  VecGetArrayRead(f, &fvals) >> ablate::checkError;
  VecGetArray(rhs, &vals) >> ablate::checkError;
  for (i = 0; i < nCells; ++i) {
    DMPlexPointLocalFieldRead(dm, lst[i], field->id, fvals, &vals[i]) >> checkError;
  }
  VecRestoreArrayRead(f, &fvals) >> ablate::checkError;
  VecRestoreArray(rhs, &vals) >> ablate::checkError;

  MatSolve(A, rhs, weights) >> ablate::checkError;

  VecDestroy(&rhs) >> ablate::checkError;

  // Now do the actual interpolation

  // Get the cell center
  DMPlexComputeCellGeometryFVM(dm, c, NULL, x0, NULL) >> ablate::checkError;

  PetscInt p1 = RBF::polyOrder + 1, dim = subDomain->GetDimensions();
  PetscInt px, py, pz;
  PetscReal *xp;
  PetscMalloc1(dim*p1, &xp) >> ablate::checkError;

  for (PetscInt d = 0; d < dim; ++d) {
    x0[d] = xEval[d] - x0[d]; // Shifted center

    // precompute powers
    xp[d*p1 + 0] = 1.0;
    for (px = 1; px < p1; ++px) {
      xp[d*p1 + px] = xp[d*p1 + (px-1)]*x0[d];
    }
  }


  PetscReal   interpVal = 0.0;
  VecGetArray(weights, &vals) >> ablate::checkError;
  for (i = 0; i < nCells; ++i) {
    interpVal += vals[i]*RBFVal(x0, &x[i*dim]);
  }

  // Augmented polynomial contributions
  if (dim == 2) {
    for (py = 0; py < p1; ++py) {
      for (px = 0; px < p1-py; ++px) {
        interpVal += vals[i++] * xp[0*p1 + px] * xp[1*p1 + py];
      }
    }
  } else {
    for (pz = 0; pz < p1; ++pz) {
      for (py = 0; py < p1-pz; ++py) {
        for (px = 0; px < p1-py-pz; ++px ){
          interpVal += vals[i++] * xp[0*p1 + px] * xp[1*p1 + py] * xp[2*p1 + pz];
        }
      }
    }
  }
  VecRestoreArray(weights, &vals) >> ablate::checkError;
  VecDestroy(&weights) >> ablate::checkError;
  PetscFree(xp) >> ablate::checkError;

  return interpVal;

}



/************ End Interpolation Code **********************/


/************ Constructor, Setup, and Initialization Code **********************/



// Return the range of DMPlex objects at a given depth in a subDomain and region. This is pulled from ablate::solver::Solver::GetRange
//    which already has the subDomain and region information.
//    Note: This seems like it should be in ablate::domain and the solver will call this. Need to talk to Matt. M about it.
void RBF::GetRange(std::shared_ptr<ablate::domain::SubDomain> subDomain, const std::shared_ptr<ablate::domain::Region> region, PetscInt depth, ablate::solver::Range &range) const {
    // Start out getting all the points
    IS allPointIS;
    DMGetStratumIS(subDomain->GetDM(), "dim", depth, &allPointIS) >> checkError;
    if (!allPointIS) {
        DMGetStratumIS(subDomain->GetDM(), "depth", depth, &allPointIS) >> checkError;
    }

    // If there is a label for this solver, get only the parts of the mesh that here
    if (region) {
        DMLabel label;
        DMGetLabel(subDomain->GetDM(), region->GetName().c_str(), &label);

        IS labelIS;
        DMLabelGetStratumIS(label, region->GetValue(), &labelIS) >> checkError;
        ISIntersect_Caching_Internal(allPointIS, labelIS, &range.is) >> checkError;
        ISDestroy(&labelIS) >> checkError;
    } else {
        PetscObjectReference((PetscObject)allPointIS) >> checkError;
        range.is = allPointIS;
    }

    // Get the point range
    if (range.is == nullptr) {
        // There are no points in this region, so skip
        range.start = 0;
        range.end = 0;
        range.points = nullptr;
    } else {
        // Get the range
        ISGetPointRange(range.is, &range.start, &range.end, &range.points) >> checkError;
    }

    // Clean up the allCellIS
    ISDestroy(&allPointIS) >> checkError;
}


void RBF::GetCellRange(std::shared_ptr<ablate::domain::SubDomain> subDomain, const std::shared_ptr<ablate::domain::Region> region, ablate::solver::Range &cellRange) const {
    // Start out getting all the cells
    PetscInt depth;
    DMPlexGetDepth(subDomain->GetDM(), &depth) >> checkError;
    RBF::GetRange(subDomain, region, depth, cellRange);
}


void RBF::RestoreRange(ablate::solver::Range &range) const {
    if (range.is) {
        ISRestorePointRange(range.is, &range.start, &range.end, &range.points) >> checkError;
        ISDestroy(&range.is) >> checkError;
    }
}

RBF::RBF(PetscInt polyOrder, bool hasDerivatives, bool hasInterpolation) :
    polyOrder(polyOrder == 0 ? __RBF_DEFAULT_POLYORDER : polyOrder),
    hasDerivatives(hasDerivatives),
    hasInterpolation(hasInterpolation) {}


RBF::~RBF() {}

// This is done once
void RBF::Setup(std::shared_ptr<ablate::domain::SubDomain> subDomain) {

  if ((!RBF::hasDerivatives) && (!RBF::hasInterpolation)) {
    throw std::runtime_error("ablate::domain::RBF requires either derivatives or interpolation.");
  }

  RBF::subDomain = subDomain;

  PetscInt dim = subDomain->GetDimensions();

  // The number of polynomial values is (p+2)(p+1)/2 in 2D and (p+3)(p+2)(p+1)/6 in 3D
  PetscInt p = RBF::polyOrder;
  if (dim == 2) {
    RBF::nPoly = (p+2)*(p+1)/2;
  } else {
    RBF::nPoly = (p+3)*(p+2)*(p+1)/6;
  }

  // Set the minimum number of cells to get compute the RBF matrix
  RBF::minNumberCells = (PetscInt)floor(2*(RBF::nPoly));

  if (RBF::hasDerivatives) {

    // Now setup the derivatives required for curvature/normal calculations. This should probably move over to user-option
    PetscInt nDer = 0;
    PetscInt dx[10], dy[10], dz[10];

    nDer = ( dim == 2 ) ? 5 : 10;
    PetscInt i = 0;
    dx[i] = 1; dy[i] = 0; dz[i++] = 0;
    dx[i] = 0; dy[i] = 1; dz[i++] = 0;
    dx[i] = 2; dy[i] = 0; dz[i++] = 0;
    dx[i] = 0; dy[i] = 2; dz[i++] = 0;
    dx[i] = 1; dy[i] = 1; dz[i++] = 0;
    if( dim == 3) {
      dx[i] = 0; dy[i] = 0; dz[i++] = 1;
      dx[i] = 0; dy[i] = 0; dz[i++] = 2;
      dx[i] = 1; dy[i] = 0; dz[i++] = 1;
      dx[i] = 0; dy[i] = 1; dz[i++] = 1;
      dx[i] = 1; dy[i] = 1; dz[i++] = 1;
    }
    SetDerivatives(nDer, dx, dy, dz);
  }


}

void RBF::Initialize(solver::Range cellRange) {

  // If this is called due to a grid change then release the old memory. In this case cEnd - cStart will be greater than zero.
  if ((RBF::cEnd - RBF::cStart) > 0) {
    for (PetscInt c = RBF::cStart; c < RBF::cEnd; ++c ){
      PetscFree(RBF::stencilList[c]);
      if(RBF::RBFMatrix[c]) MatDestroy(&(RBF::RBFMatrix[c]));
      PetscFree(RBF::stencilWeights);
      PetscFree(RBF::stencilXLocs[c]);
    }
    PetscFree6(RBF::cellList, RBF::nStencil, RBF::stencilList, RBF::RBFMatrix, RBF::stencilXLocs, RBF::stencilWeights) >> ablate::checkError;
  }

  RBF::cStart = cellRange.start;
  RBF::cEnd   = cellRange.end;

  // Both interpolation and derivatives need the list of points
  PetscInt nCells = RBF::cEnd - RBF::cStart;
  PetscMalloc6(nCells, &(RBF::cellList), nCells, &(RBF::nStencil), nCells, &(RBF::stencilList), nCells, &(RBF::RBFMatrix), nCells, &(RBF::stencilXLocs), nCells, &(RBF::stencilWeights)) >> ablate::checkError;

  // Shift so that we can use cell range directly
  RBF::cellList -= cStart;
  RBF::nStencil -= cStart;
  RBF::stencilList -= cStart;
  RBF::RBFMatrix -= cStart;
  RBF::stencilXLocs -= cStart;
  RBF::stencilWeights -= cStart;

  for (PetscInt c = cStart; c < cEnd; ++c) {

    RBF::cellList[c] = cellRange.points ? cellRange.points[c] : c;

    RBF::nStencil[c] = -1;
    RBF::stencilList[c] = nullptr;

    RBF::RBFMatrix[c] = nullptr;
    RBF::stencilXLocs[c] = nullptr;
    RBF::stencilWeights[c] = nullptr;
  }

}

