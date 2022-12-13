//#include "rbf.hpp"
//#include "utilities/petscError.hpp"
//#include "domain/subDomain.hpp"


//static const std::map<std::string, ablate::radialBasis::RBF::RBFType> stringToRBFType = {{"", ablate::radialBasis::RBF::RBFType::MQ},
//                                                                                           {"phs", ablate::radialBasis::RBF::RBFType::PHS},
//                                                                                           {"PHS", ablate::radialBasis::RBF::RBFType::PHS},
//                                                                                           {"mq", ablate::radialBasis::RBF::RBFType::MQ},
//                                                                                           {"MQ", ablate::radialBasis::RBF::RBFType::MQ},
//                                                                                           {"imq", ablate::radialBasis::RBF::RBFType::IMQ},
//                                                                                           {"IMQ", ablate::radialBasis::RBF::RBFType::IMQ},
//                                                                                           {"ga", ablate::radialBasis::RBF::RBFType::GA},
//                                                                                           {"GA", ablate::radialBasis::RBF::RBFType::GA}};

//std::istream& ablate::radialBasis::operator>>(std::istream& is, ablate::radialBasis::RBF::RBFType& v) {
//    std::string enumString;
//    is >> enumString;
//    v = stringToRBFType.at(enumString);
//    return is;
//}


//using namespace ablate::radialBasis;

///************ Begin Base Class **********************/
//// Setup for the RBF class
//RBF::RBF(DM dm, PetscInt p){

//  RBF::dm = dm;   // Set the DM
//  RBF::p = p;     // The augmented polynomial order
//  DMGetDimension(dm, &(RBF::dim)) >> ablate::checkError; // Get the dimension of the problem

//  // The number of polynomial values is (p+2)(p+1)/2 in 2D and (p+3)(p+2)(p+1)/6 in 3D
//  if (dim == 2) {
//    RBF::nPoly = (p+2)*(p+1)/2;
//  } else {
//    RBF::nPoly = (p+3)*(p+2)*(p+1)/6;
//  }

//  // Set the minimum number of cells to get compute the RBF matrix
//  RBF::minNumberCells = (PetscInt)floor(2*(RBF::nPoly));

//  DMPlexGetHeightStratum(RBF::dm, 0, &(RBF::cStart), &(RBF::cEnd)) >> ablate::checkError;      // Range of cells



//  // Both interpolation and derivatives need the list of points
//  PetscInt n = RBF::cEnd - RBF::cStart;
//  PetscMalloc2(n, &(RBF::nStencil), n, &(RBF::stencilList)) >> ablate::checkError;
//  PetscMalloc2(n, &(RBF::RBFMatrix), n, &(RBF::stencilXLocs));

//  // Offset the indices so that we can use stratum numbering
//  RBF::nStencil -= cStart;
//  RBF::stencilList -= cStart;


//  RBF::RBFMatrix -= cStart;
//  RBF::stencilXLocs -= cStart;

//  for (PetscInt c = cStart; c < cEnd; ++c) {
//    RBF::nStencil[c] = -1;
//    RBF::stencilList[c] = nullptr;

//    RBF::RBFMatrix[c] = nullptr;
//    RBF::stencilXLocs[c] = nullptr;
//  }

//}

//RBF::~RBF(){

//  PetscInt c;

//  for (c = RBF::cStart; c < RBF::cEnd; ++c) {
//    PetscFree(RBF::stencilList[c]);
//  }

//  PetscFree2(RBF::nStencil, RBF::stencilList) >> ablate::checkError;

//  if (RBF::hasDerivativeInformation) {
//    PetscFree2(RBF::stencilWeights, RBF::dxyz) >> ablate::checkError;
//  }


//  for (c = RBF::cStart; c < RBF::cEnd; ++c) {
//    if (RBF::RBFMatrix[c]) {
//      MatDestroy(&(RBF::RBFMatrix[c]));
//    }
//    PetscFree(RBF::stencilXLocs[c]);
//  }
//  PetscFree2(RBF::RBFMatrix, RBF::stencilXLocs);

//}


//void RBF::ShowParameters(){
//  PetscPrintf(PETSC_COMM_WORLD, "RBF Parameters\n");
//  PetscPrintf(PETSC_COMM_WORLD, "%12s: %d\n", "dim", RBF::dim);
//  PetscPrintf(PETSC_COMM_WORLD, "%12s: %d\n", "Poly Order", RBF::p);
//  PetscPrintf(PETSC_COMM_WORLD, "%12s: %d\n", "Has DM", RBF::dm!=nullptr);
//  PetscPrintf(PETSC_COMM_WORLD, "%12s: %d\n", "Min # Cells", RBF::minNumberCells);
//  PetscPrintf(PETSC_COMM_WORLD, "%12s: %d\n", "Use Vertices", RBF::useVertices);
//  if ( RBF::hasDerivativeInformation ){
//      PetscPrintf(PETSC_COMM_WORLD, "%12s: %d\n", "nDer", RBF::nDer);
//      for (PetscInt i = 0; i < RBF::nDer; ++i) {
//          PetscPrintf(PETSC_COMM_WORLD, "%12s: %d, %d, %d\n", "dx,dy,dz", RBF::dxyz[i*3+0],RBF::dxyz[i*3+1],RBF::dxyz[i*3+2]);
//      }
//  }
//}

//// Distance between two points
//PetscReal RBF::DistanceSquared(PetscReal x[], PetscReal y[]){
//  PetscInt  dim = RBF::dim;
//  PetscReal r = 0.0;
//  for (PetscInt d = 0; d < dim; ++d) {
//    r += (x[d] - y[d]) * (x[d] - y[d]);
//  }

//  return r;
//}

//// Distance between point and the origin
//PetscReal RBF::DistanceSquared(PetscReal x[]){
//  PetscInt  dim = RBF::dim;
//  PetscReal r = 0.0;
//  for (PetscInt d = 0; d < dim; ++d) {
//    r += x[d] * x[d];
//  }
//  return r;
//}

//static PetscInt fac[11] =  {1,1,2,6,24,120,720,5040,40320,362880,3628800}; // Pre-computed factorials


//// Compute the LU-factorization of the augmented RBF matrix
//// c - The center cell
//// nCells - The number of cells in the neighbor list
//// list - The list of neighbor cells
//// xCenters - Shifted centers of all the cells (output)
//// A - The LU-factorization of the augmented RBF matrix (output)
//void RBF::Matrix(PetscInt c, PetscReal **xCenters, Mat *LUA) {

//  PetscInt        dim = RBF::dim;
//  PetscInt        i, j, d;
//  PetscInt        px, py, pz, p1 = p+1;
//  PetscInt        nPoly = RBF::nPoly, matSize;
//  PetscInt        nCells, *list;
//  Mat             A;
//  PetscScalar     *vals;
//  PetscReal       *x;
//  PetscInt        p = RBF::p;   // The supplementary polynomial order
//  DM              dm = RBF::dm; // The underlying mesh

//  PetscReal       x0[dim];        // Center of the cell of interest
//  PetscReal       *xp; // Powers of the cell centers

//  // Get the list of neighbor cells
//  DMPlexGetNeighborCells(RBF::dm, c, -1, -1.0, RBF::minNumberCells, RBF::useVertices, &nCells, &list);
//  RBF::nStencil[c] = nCells;
//  RBF::stencilList[c] = list;

//  PetscMalloc1(nCells*dim*p1, &xp) >> ablate::checkError;


//  if(nPoly>=nCells){
//    throw std::invalid_argument("Number of surrounding cells, " + std::to_string(nCells) + ", can not support a requested polynomial order of " + std::to_string(p) + " which requires " + std::to_string(nPoly) + " number of cells.");
//  }

//  // Get the cell center
//  DMPlexComputeCellGeometryFVM(dm, c, NULL, x0, NULL) >> ablate::checkError;

//  // Shifted cell-centers of neighbor cells
//  PetscMalloc1(nCells*dim, &x);
//  for (i = 0; i < nCells; ++i) {
//    DMPlexComputeCellGeometryFVM(dm, list[i], NULL, &x[i*dim], NULL) >> ablate::checkError;
//    for (d = 0; d < dim; ++d) {
//      x[i*dim+d] -= x0[d];
//      // Precompute the powers for later use
//      xp[(i*dim+d)*p1 + 0] = 1.0;
//      for (px = 1; px < p+1; ++px) {
//        xp[(i*dim+d)*p1 + px] = xp[(i*dim+d)*p1 + (px-1)]*x[i*dim+d];
//      }
//    }
//  }

//  matSize = nCells + nPoly;

//  // Create the matrix
//  MatCreateSeqDense(PETSC_COMM_SELF, matSize, matSize, NULL, &A) >> ablate::checkError;
//  PetscObjectSetName((PetscObject)A,"ablate::radialBasis::RBF::A") >> ablate::checkError;
//  MatZeroEntries(A) >> ablate::checkError;
//  MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE) >> ablate::checkError;

//  MatDenseGetArrayWrite(A, &vals) >> ablate::checkError;

//  // RBF contributions to the matrix
//  for (i = 0; i < nCells; ++i) {
//    for (j = i; j < nCells; ++j) {
//      vals[i*matSize + j] = vals[j*matSize + i] = RBFVal(&x[i*dim], &x[j*dim]);
//    }
//  }

//  // Augmented polynomial contributions
//  if (dim == 2) {
//    for (i = 0; i < nCells; ++i) {
//      j = nCells;
//      for (py = 0; py < p1; ++py) {
//        for (px = 0; px < p1-py; ++px) {
//          vals[i*matSize + j] = vals[j*matSize + i] = xp[(i*dim+0)*p1 + px] * xp[(i*dim+1)*p1 + py];
//          j++;
//        }
//      }
//    }
//  } else {
//    for (i = 0; i < nCells; ++i) {
//      j = nCells;
//      for (pz = 0; pz < p1; ++pz) {
//        for (py = 0; py < p1-pz; ++py) {
//          for (px = 0; px < p1-py-pz; ++px ){
//            vals[i*matSize + j] = vals[j*matSize + i] = xp[(i*dim+0)*p1 + px] * xp[(i*dim+1)*p1 + py] * xp[(i*dim+2)*p1 + pz];
//            ++j;
//          }
//        }
//      }
//    }
//  }
//  MatDenseRestoreArrayWrite(A, &vals) >> ablate::checkError;
//  MatViewFromOptions(A,NULL,"-ablate::radialBasis::RBF::A_view") >> ablate::checkError;

//  // Factor the matrix
//  MatLUFactor(A, NULL, NULL, NULL) >> ablate::checkError;

//  PetscFree(xp) >> ablate::checkError;

//  // Assign output
//  *xCenters = x;
//  *LUA = A;

//}


///************ Begin Derivative Code **********************/

//// Compute the RBF weights at the cell center of p using a cell-list
//// c - The center cell
//// nCells - The number of cells in the neighbor list
//// list - The list of neighbor cells
//// nDer - The number of derviatives to compute
//// dx, dy, dz - Arrays of length nDer. The derivative is (dx(i), dy(i), dz(i))
//// weights - The resulting weights. Array of length nCells.
//void RBF::SetupDerivativeStencils(PetscInt c) {

//  Mat       A = RBF::RBFMatrix[c], B = nullptr;
//  PetscReal *x = nullptr;   // Shifted cell-centers of the neigbor list
//  // This also computes nStencil[c]
//  if (A == nullptr) {
//    RBF::Matrix(c, &x, &A);
//  }

//  PetscInt    dim = RBF::dim;
//  PetscInt    nCells = RBF::nStencil[c];
//  PetscInt    matSize = nCells + RBF::nPoly;
//  PetscInt    nDer = RBF::nDer;
//  PetscInt    *dxyz = RBF::dxyz;

//  PetscInt    i, j;
//  PetscInt    px, py, pz, p1 = p+1;
//  PetscScalar *vals = nullptr;

//  //Create the RHS
//  MatCreateSeqDense(PETSC_COMM_SELF, matSize, nDer, NULL, &B) >> ablate::checkError;
//  PetscObjectSetName((PetscObject)B,"ablate::radialBasis::RBF::rhs") >> ablate::checkError;
//  MatZeroEntries(B) >> ablate::checkError;
//  MatDenseGetArrayWrite(B, &vals) >> ablate::checkError;

//  // Derivatives of the RBF
//  for (i = 0; i < nCells; ++i) {
//    for (j = 0; j < nDer; ++j) {
//      vals[i + j*matSize] = RBFDer(&x[i*dim], dxyz[j*3 + 0], dxyz[j*3 + 1], dxyz[j*3 + 2]);
//    }
//  }

//  // Derivatives of the augmented polynomials
//  if (dim == 2) {
//    for (j = 0; j < nDer; ++j) {
//      i = nCells;
//      for (py = 0; py < p1; ++py) {
//        for (px = 0; px < p1-py; ++px ){
//          if(dxyz[j*3 + 0] == px && dxyz[j*3 + 1] == py) {
//            vals[i + j*matSize] = fac[px]*fac[py];
//          }
//          ++i;
//        }
//      }

//    }
//  } else {
//    for (j = 0; j < nDer; ++j) {
//      i = nCells;
//      for (pz = 0; pz < p1; ++pz) {
//        for (py = 0; py < p1-pz; ++py) {
//          for (px = 0; px < p1-py-pz; ++px ){
//            if(dxyz[j*3 + 0] == px && dxyz[j*3 + 1] == py && dxyz[j*3 + 2] == pz) {
//              vals[i + j*matSize] = fac[px]*fac[py]*fac[pz];
//            }
//            ++i;
//          }
//        }
//      }
//    }
//  }

//  MatDenseRestoreArrayWrite(B, &vals) >> ablate::checkError;

//  MatViewFromOptions(B,NULL,"-ablate::radialBasis::RBF::rhs_view") >> ablate::checkError;

//  MatMatSolve(A, B, B) >> ablate::checkError;

//  MatViewFromOptions(B,NULL,"-ablate::radialBasis::RBF::sol_view") >> ablate::checkError;

//  // Now populate the output
//  PetscMalloc1(nDer*nCells, &(RBF::stencilWeights[c])) >> ablate::checkError;
//  PetscReal *wt = RBF::stencilWeights[c];
//  MatDenseGetArrayWrite(B, &vals) >> ablate::checkError;
//  for (i = 0; i < nCells; ++i) {
//    for (j = 0; j < nDer; ++j) {
//      wt[i*nDer + j] = vals[i + j*matSize];
//    }
//  }
//  MatDenseGetArrayWrite(B, &vals) >> ablate::checkError;

//  if (RBF::hasInterpolation) {
//    RBF::RBFMatrix[c] = A;
//    RBF::stencilXLocs[c] = x;
//  }
//  else {
//    MatDestroy(&A) >> ablate::checkError;
//    PetscFree(x) >> ablate::checkError;
//  }
//  MatDestroy(&B) >> ablate::checkError;

//}



//void RBF::SetupDerivativeStencils() {
//  PetscInt c, cStart = RBF::cStart, cEnd = RBF::cEnd;

//  for (c = cStart; c < cEnd; ++c) {
//    RBF::SetupDerivativeStencils(c);
//  }
//}


//// Set the derivatives to use
//// nDer - Number of derivatives to set
//// dx, dy, dz - Lists of length nDer indicating the derivatives
//void RBF::SetDerivatives(PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[], PetscBool useVertices){

//  if (nDer > 0) {

//    PetscInt c, cStart = RBF::cStart, cEnd = RBF::cEnd, n;

//    RBF::hasDerivativeInformation = PETSC_TRUE;
//    RBF::useVertices = useVertices;
//    RBF::nDer = nDer;

//    n = cEnd - cStart;

//    PetscMalloc2(n, &(RBF::stencilWeights), 3*nDer, &(RBF::dxyz)) >> ablate::checkError;

//    // Offset the indices so that we can use stratum numbering
//    RBF::stencilWeights -= cStart;

//    for (c = cStart; c < cEnd; ++c) {
//      RBF::stencilWeights[c] = nullptr;
//    }

//    // Store the derivatives
//    for (c = 0; c < nDer; ++c) {
//      RBF::dxyz[c*3 + 0] = dx[c];
//      RBF::dxyz[c*3 + 1] = dy[c];
//      RBF::dxyz[c*3 + 2] = dz[c];
//    }
//  }


//}

//// Set derivatives, defaulting to using vertices
//void RBF::SetDerivatives(PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[]){
//  RBF::SetDerivatives(nDer, dx, dy, dz, PETSC_TRUE);
//}


//// Return the requested derivative
//// f - Vector containing the data
//// c - Location to evaluate at
//// dx, dy, dz - The derivatives
//PetscReal RBF::EvalDer(Vec f, PetscInt c, PetscInt dx, PetscInt dy, PetscInt dz){

//  PetscInt  derID = -1, nDer = RBF::nDer, *dxyz = RBF::dxyz;
//  PetscReal val = 0.0, *array;

//  // Search for the particular index. Probably need to do something different in the future to avoid re-doing the same calculation many times
//  while ((dxyz[++derID*3 + 0] != dx || dxyz[derID*3 + 1] != dy || dxyz[derID*3 + 2] != dz) && derID<nDer){ }

//  if (derID==nDer) {
//    throw std::invalid_argument("Derivative of (" + std::to_string(dx) + ", " + std::to_string(dy) + ", " + std::to_string(dz) + ") is not setup.");
//  }

//  // If the stencil hasn't been setup yet do so
//  if (RBF::nStencil[c] < 1) {
//    RBF::SetupDerivativeStencils(c);
//  }

//  PetscReal *wt = RBF::stencilWeights[c];
//  PetscInt  cStart = RBF::cStart, nStencil = RBF::nStencil[c], *lst = RBF::stencilList[c];

//  VecGetArray(f, &array) >> ablate::checkError;


//  for (PetscInt i = 0; i < nStencil; ++i) {
//    val += wt[i*nDer + derID]*array[lst[i] - cStart];
//  }

//  VecRestoreArray(f, &array) >> ablate::checkError;

//  return val;
//}




////// Return the requested derivative
////// f - Vector containing the data
////// c - Location to evaluate at
////// dx, dy, dz - The derivatives
////PetscReal RBF::EvalDer(ablate::domain::Field field, PetscInt c, PetscInt dx, PetscInt dy, PetscInt dz){

////This should change the Vec to a FieldID. Then use DMPlexPointGlobalFieldRef/Read to get the values


////  PetscInt  derID = -1, nDer = RBF::nDer, *dxyz = RBF::dxyz;
////  PetscReal val = 0.0, *array;

////  // Search for the particular index. Probably need to do something different in the future to avoid re-doing the same calculation many times
////  while ((dxyz[++derID*3 + 0] != dx || dxyz[derID*3 + 1] != dy || dxyz[derID*3 + 2] != dz) && derID<nDer){ }

////  if (derID==nDer) {
////    throw std::invalid_argument("Derivative of (" + std::to_string(dx) + ", " + std::to_string(dy) + ", " + std::to_string(dz) + ") is not setup.");
////  }

////  // If the stencil hasn't been setup yet do so
////  if (RBF::nStencil[c] < 1) {
////    RBF::SetupDerivativeStencils(c);
////  }

////  PetscReal *wt = RBF::stencilWeights[c];
////  PetscInt  cStart = RBF::cStart, nStencil = RBF::nStencil[c], *lst = RBF::stencilList[c];

////  VecGetArray(f, &array) >> ablate::checkError;


////  for (PetscInt i = 0; i < nStencil; ++i) {
////    val += wt[i*nDer + derID]*array[lst[i] - cStart];
////  }

////  VecRestoreArray(f, &array) >> ablate::checkError;

////  return val;

////}


///************ End Derivative Code **********************/




///************ Begin Interpolation Code **********************/

//void RBF::SetInterpolation(PetscBool hasInterpolation) {
//  RBF::hasInterpolation = hasInterpolation;
//}

//PetscReal RBF::Interpolate(Vec f, PetscInt c, PetscReal xEval[3]) {

//  Mat         A = RBF::RBFMatrix[c];
//  Vec         weights, rhs;
//  PetscInt    nCells, *lst;
//  PetscInt    i, cStart = RBF::cStart;
//  PetscScalar *vals, *fvals;
//  PetscReal   *x = RBF::stencilXLocs[c], x0[3];

//  if (A == nullptr) {
//    RBF::Matrix(c, &x, &A);
//    RBF::RBFMatrix[c] = A;
//    RBF::stencilXLocs[c] = x;
//  }

//  nCells = RBF::nStencil[c];
//  lst = RBF::stencilList[c];

//  MatCreateVecs(A, &weights, &rhs) >> ablate::checkError;
//  VecZeroEntries(weights) >> ablate::checkError;
//  VecZeroEntries(rhs) >> ablate::checkError;

//  // The function values
//  VecGetArray(f, &fvals) >> ablate::checkError;
//  VecGetArray(rhs, &vals) >> ablate::checkError;
//  for (i = 0; i < nCells; ++i) {
//    vals[i] = fvals[lst[i] - cStart];
//  }
//  VecRestoreArray(f, &fvals) >> ablate::checkError;
//  VecRestoreArray(rhs, &vals) >> ablate::checkError;

//  MatSolve(A, rhs, weights) >> ablate::checkError;

//  VecDestroy(&rhs) >> ablate::checkError;

//  // Now do the actual interpolation

//  // Get the cell center
//  DMPlexComputeCellGeometryFVM(RBF::dm, c, NULL, x0, NULL) >> ablate::checkError;

//  PetscInt p1 = RBF::p + 1, dim = RBF::dim;
//  PetscInt px, py, pz;
//  PetscReal *xp;
//  PetscMalloc1(dim*p1, &xp) >> ablate::checkError;

//  for (PetscInt d = 0; d < dim; ++d) {
//    x0[d] = xEval[d] - x0[d]; // Shifted center

//    // precompute powers
//    xp[d*p1 + 0] = 1.0;
//    for (px = 1; px < p+1; ++px) {
//      xp[d*p1 + px] = xp[d*p1 + (px-1)]*x0[d];
//    }
//  }


//  PetscReal   interpVal = 0.0;
//  VecGetArray(weights, &vals) >> ablate::checkError;
//  for (i = 0; i < nCells; ++i) {
//    interpVal += vals[i]*RBFVal(x0, &x[i*dim]);
//  }

//  // Augmented polynomial contributions
//  if (dim == 2) {
//    for (py = 0; py < p1; ++py) {
//      for (px = 0; px < p1-py; ++px) {
//        interpVal += vals[i++] * xp[0*p1 + px] * xp[1*p1 + py];
//      }
//    }
//  } else {
//    for (pz = 0; pz < p1; ++pz) {
//      for (py = 0; py < p1-pz; ++py) {
//        for (px = 0; px < p1-py-pz; ++px ){
//          interpVal += vals[i++] * xp[0*p1 + px] * xp[1*p1 + py] * xp[2*p1 + pz];
//        }
//      }
//    }
//  }
//  VecRestoreArray(weights, &vals) >> ablate::checkError;
//  VecDestroy(&weights) >> ablate::checkError;
//  PetscFree(xp) >> ablate::checkError;

//  return interpVal;

//}



///************ End Interpolation Code **********************/



///************ End Base Class **********************/




///************ Begin Polyharmonic Spline Derived Class **********************/
//PHS::PHS(DM dm, PetscInt p, PetscInt m) : RBF(std::move(dm), std::move(p)) {
//  PHS::phsOrder = m;
//}

// //Polyharmonic spline: r^m
//PetscReal PHS::InternalVal(PetscReal x[], PetscReal y[]) {
//  PetscInt  m = PHS::phsOrder;   // The PHS order
//  PetscReal r = PHS::DistanceSquared(x, y);

//  return PetscPowReal(r, 0.5*((PetscReal)m));
//}

//// Derivatives of Polyharmonic spline at a location.
//PetscReal PHS::InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) {
//  PetscInt  m = PHS::phsOrder;   // The PHS order
//  PetscReal r = DistanceSquared(x);

//  r = PetscSqrtReal(r);

//  switch (dx + 10*dy + 100*dz) {
//    case 0:
//      r = PetscPowReal(r, (PetscReal)m);
//      break;
//    case 1: // x
//      r = -m*x[0]*PetscPowReal(r, (PetscReal)(m-2));
//      break;
//    case 2: // xx
//      r = m*PetscPowReal(r, (PetscReal)(m-2)) + m*(m-2)*x[0]*x[0]*PetscPowReal(r, (PetscReal)(m-4));
//      break;
//    case 10: // y
//      r = -m*x[1]*PetscPowReal(r, (PetscReal)(m-2));
//      break;
//    case 20: // yy
//      r = m*PetscPowReal(r, (PetscReal)(m-2)) + m*(m-2)*x[1]*x[1]*PetscPowReal(r, (PetscReal)(m-4));
//      break;
//    case 100: // z
//      r = -m*x[2]*PetscPowReal(r, (PetscReal)(m-2));
//      break;
//    case 200: // zz
//      r = m*PetscPowReal(r, (PetscReal)(m-2)) + m*(m-2)*x[2]*x[2]*PetscPowReal(r, (PetscReal)(m-4));
//      break;
//    case 11: // xy
//      r = m*(m-2)*x[0]*x[1]*PetscPowReal(r, (PetscReal)(m-4));
//      break;
//    case 101: // xz
//      r = m*(m-2)*x[0]*x[2]*PetscPowReal(r, (PetscReal)(m-4));
//      break;
//    case 110: // yz
//      r = m*(m-2)*x[1]*x[2]*PetscPowReal(r, (PetscReal)(m-4));
//      break;
//    default:
//      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown derivative!\n");
//  }

//  return r;
//}
///************ End Polyharmonic Spline Derived Class **********************/


///************ Begin Multiquadric Derived Class **********************/
//MQ::MQ(DM dm, PetscInt p, PetscReal scale) : RBF(std::move(dm), std::move(p)) {
//  MQ::scale = scale;
//}

//// Multiquadric: sqrt(1+(er)^2)
//PetscReal MQ::InternalVal(PetscReal x[], PetscReal y[]) {

//  PetscReal h = MQ::scale;
//  PetscReal e = 1.0/h;
//  PetscReal r = DistanceSquared(x, y);

//  return PetscSqrtReal(1.0 + e*e*r);
//}

//// Derivatives of Multiquadric spline at a location.
//PetscReal MQ::InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) {

//  PetscReal h = MQ::scale;
//  PetscReal e = 1.0/h;
//  PetscReal r = DistanceSquared(x);

//  r = PetscSqrtReal(1.0 + e*e*r);

//  switch (dx + 10*dy + 100*dz) {
//    case 0:
//      // Do nothing
//      break;
//    case 1: // x
//      r = -e*e*x[0]/r;
//      break;
//    case 2: // xx
//      r = e*e*(1.0 + e*e*(x[1]*x[1] + x[2]*x[2]))/PetscPowReal(r, 3.0);
//      break;
//    case 10: // y
//      r = -e*e*x[1]/r;
//      break;
//    case 20: // yy
//      r = e*e*(1.0 + e*e*(x[0]*x[0] + x[2]*x[2]))/PetscPowReal(r, 3.0);
//      break;
//    case 100: // z
//      r = -e*e*x[2]/r;
//      break;
//    case 200: // zz
//      r = e*e*(1.0 + e*e*(x[0]*x[0] + x[1]*x[1]))/PetscPowReal(r, 3.0);
//      break;
//    case 11: // xy
//      r = -PetscSqr(e*e)*x[0]*x[1]/PetscPowReal(r, 3.0);
//      break;
//    case 101: // xz
//      r = -PetscSqr(e*e)*x[0]*x[2]/PetscPowReal(r, 3.0);
//      break;
//    case 110: // yz
//      r = -PetscSqr(e*e)*x[1]*x[2]/PetscPowReal(r, 3.0);
//      break;
//    default:
//      throw std::invalid_argument("Derivative of (" + std::to_string(dx) + ", " + std::to_string(dy) + ", " + std::to_string(dz) + ") is not setup.");
//  }

//  return r;
//}

///************ End Multiquadric Derived Class **********************/

///************ Begin Inverse Multiquadric Derived Class **********************/
//IMQ::IMQ(DM dm, PetscInt p, PetscReal scale) : RBF(std::move(dm), std::move(p)) {
//  IMQ::scale = scale;
//}

//// Multiquadric: sqrt(1+(er)^2)
//PetscReal IMQ::InternalVal(PetscReal x[], PetscReal y[]) {

//  PetscReal h = IMQ::scale;
//  PetscReal e = 1.0/h;
//  PetscReal r = DistanceSquared(x, y);

//  return 1.0/PetscSqrtReal(1.0 + e*e*r);
//}

//// Derivatives of Multiquadric spline at a location.
//PetscReal IMQ::InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) {

//  PetscReal h = IMQ::scale;
//  PetscReal e = 1.0/h;
//  PetscReal r = DistanceSquared(x);

//  r = PetscSqrtReal(1.0 + e*e*r);

//  switch (dx + 10*dy + 100*dz) {
//    case 0:
//      r = 1.0/r;
//      break;
//    case 1: // x
//      r = -e*e*x[0]/PetscPowReal(r, 3.0);
//      break;
//    case 2: // xx
//      r = -e*e*(1.0 + e*e*(-2.0*x[0]*x[0] + x[1]*x[1] + x[2]*x[2]))/PetscPowReal(r, 5.0);
//      break;
//    case 10: // y
//      r = -e*e*x[1]/PetscPowReal(r, 3.0);
//      break;
//    case 20: // yy
//      r = -e*e*(1.0 + e*e*(x[0]*x[0] - 2.0*x[1]*x[1] + x[2]*x[2]))/PetscPowReal(r, 5.0);
//      break;
//    case 100: // z
//      r = -e*e*x[2]/PetscPowReal(r, 3.0);
//      break;
//    case 200: // zz
//      r = -e*e*(1.0 + e*e*(x[0]*x[0] + x[1]*x[1] - 2.0*x[2]*x[2]))/PetscPowReal(r, 5.0);
//      break;
//    case 11: // xy
//      r = 3.0*PetscSqr(e*e)*x[0]*x[1]/PetscPowReal(r, 5.0);
//      break;
//    case 101: // xz
//      r = 3.0*PetscSqr(e*e)*x[0]*x[2]/PetscPowReal(r, 5.0);
//      break;
//    case 110: // yz
//      r = 3.0*PetscSqr(e*e)*x[1]*x[2]/PetscPowReal(r, 5.0);
//      break;
//    default:
//      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown derivative!\n");
//  }

//  return r;
//}
///************ End Inverse Multiquadric Derived Class **********************/

///************ Begin Gaussian Derived Class **********************/
//GA::GA(DM dm, PetscInt p, PetscReal scale) : RBF(std::move(dm), std::move(p)) {
//  GA::scale = scale;
//}

//// Gaussian: r^m
//PetscReal GA::InternalVal(PetscReal x[], PetscReal y[]) {

//  PetscReal h = GA::scale;
//  PetscReal e2 = 1.0/(h*h);
//  PetscReal r = DistanceSquared(x, y);

//  return PetscExpReal(-r*e2);
//}

//// Derivatives of Gaussian spline at a location.
//PetscReal GA::InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) {

//  PetscReal h = GA::scale;
//  PetscReal e2 = 1.0/(h*h);
//  PetscReal r = DistanceSquared(x);

//  r = PetscExpReal(-r*e2);

//  switch (dx + 10*dy + 100*dz) {
//    case 0:
//      // Do nothing
//      break;
//    case 1: // x
//      r *= -2.0*e2*x[0];
//      break;
//    case 2: // xx
//      r *= 2.0*e2*(2.0*e2*x[0]*x[0]-1.0);
//      break;
//    case 10: // x[1]
//      r *= -2.0*e2*x[1];
//      break;
//    case 20: // yy
//      r *= 2.0*e2*(2.0*e2*x[1]*x[1]-1.0);
//      break;
//    case 100: // x[2]
//      r *= -2.0*e2*x[2];
//      break;
//    case 200: // zz
//      r *= 2.0*e2*(2.0*e2*x[2]*x[2]-1.0);
//      break;
//    case 11: // xy
//      r *= 4.0*e2*e2*x[0]*x[1];
//      break;
//    case 101: // xz
//      r *= 4.0*e2*e2*x[0]*x[2];
//      break;
//    case 110: // yz
//      r *= 4.0*e2*e2*x[1]*x[2];
//      break;
//    case 111:
//      r *= 8.0*e2*e2*e2*x[1]*x[2];
//      break;
//    default:
//      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown derivative!\n");
//  }

//  return r;
//}
///************ End Gaussian Derived Class **********************/


