#include "rbf.hpp"
#include "utilities/petscError.hpp"
#include "domain/subDomain.hpp"

using namespace ablate::levelSet;

/************ Begin Base Class **********************/

// Setup for the RBF class
RBF::RBF(DM dm, PetscInt p){

  RBF::dm = dm;   // Set the DM
  RBF::p = p;     // The augmented polynomial order
  DMGetDimension(dm, &(RBF::dim)) >> ablate::checkError; // Get the dimension of the problem

  // The number of polynomial values is (p+2)(p+1)/2 in 2D and (p+3)(p+2)(p+1)/6 in 3D
  if (dim == 2) {
    RBF::nPoly = (p+2)*(p+1)/2;
  } else {
    RBF::nPoly = (p+3)*(p+2)*(p+1)/6;
  }

}


void RBF::ShowParameters(){
  PetscPrintf(PETSC_COMM_WORLD, "RBF Parameters\n");
  PetscPrintf(PETSC_COMM_WORLD, "%12s: %d\n", "dim", RBF::dim);
  PetscPrintf(PETSC_COMM_WORLD, "%12s: %d\n", "Poly Order", RBF::p);
  PetscPrintf(PETSC_COMM_WORLD, "%12s: %d\n", "Has DM", RBF::dm!=nullptr);

}

// Distance between two points
PetscReal RBF::DistanceSquared(PetscReal x[], PetscReal y[]){
  PetscInt  dim = RBF::dim;
  PetscReal r = 0.0;
  for (PetscInt d = 0; d < dim; ++d) {
    r += (x[d] - y[d]) * (x[d] - y[d]);
  }

  return r;
}

// Distance between point and the origin
PetscReal RBF::DistanceSquared(PetscReal x[]){
  PetscInt  dim = RBF::dim;
  PetscReal r = 0.0;
  for (PetscInt d = 0; d < dim; ++d) {
    r += x[d] * x[d];
  }
  return r;
}




static PetscInt fac[11] =  {1,1,2,6,24,120,720,5040,40320,362880,3628800}; // Pre-computed factorials

// Compute the RBF weights at the cell center of p using a cell-list
// c - The center cell
// nCells - The number of cells in the neighbor list
// list - The list of neighbor cells
// nDer - The number of derviatives to compute
// dx, dy, dz - Arrays of length nDer. The derivative is (dx(i), dy(i), dz(i))
// weights - The resulting weights. Array of length nCells.
void RBF::Weights(PetscInt c, PetscInt nCells, PetscInt list[], PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[], PetscReal *weights[]) {

  PetscInt        dim = RBF::dim;
  PetscInt        i, j, d;
  PetscInt        px, py, pz, p1 = p+1;
  PetscInt        nPoly = RBF::nPoly, matSize;
  Mat             A, B;
  PetscScalar     *vals;

  PetscInt        p = RBF::p;   // The supplementary polynomial order
  DM              dm = RBF::dm; // The underlying mesh

  PetscReal       x0[dim];        // Center of the cell of interest
  PetscReal       x[nCells*dim];  // Centers of all cells
  PetscReal       xp[nCells*dim*p1]; // Powers of the



//  // If the number of cells does not support the requested augmented polynomial order decrease the order. Maybe better to just throw an error?
//  while (nPoly > nCells && p > 0) {
//    --p;
//    if (dim == 2) {
//      nPoly = (p+2)*(p+1)/2;
//    } else {
//      nPoly = (p+3)*(p+2)*(p+1)/6;
//    }
//  }
//  p1 = p+1;

  if(nPoly>=nCells){
    throw std::invalid_argument("Number of surrounding cells, " + std::to_string(nCells) + ", can not support a requested polynomial order of " + std::to_string(p) + " which requires " + std::to_string(nPoly) + " number of cells.");
  }

  // Get the cell center
  DMPlexComputeCellGeometryFVM(dm, c, NULL, x0, NULL) >> ablate::checkError;

  // Shifted cell-centers of neighbor cells
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
  PetscObjectSetName((PetscObject)A,"ablate::levelSet::RBF::A") >> ablate::checkError;
  MatZeroEntries(A) >> ablate::checkError;
  MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE) >> ablate::checkError;


  MatDenseGetArrayWrite(A, &vals) >> ablate::checkError;

  // RBF contributions to the matrix
  for (i = 0; i < nCells; ++i) {
    for (j = i; j < nCells; ++j) {
      vals[i*matSize + j] = vals[j*matSize + i] = Val(&x[i*dim], &x[j*dim]);
    }
  }

  // Augmented polynomial contributions
  if (dim == 2) {
    for (i = 0; i < nCells; ++i) {
      j = nCells;
      for (py = 0; py < p1; ++py) {
        for (px = 0; px < p1-py; ++px) {
          vals[i*matSize + j] = vals[j*matSize + i] = xp[(i*dim+0)*p1 + px] * xp[(i*dim+1)*p1 + py];
          j++;
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
  MatViewFromOptions(A,NULL,"-ablate::levelSet::RBF::A_view") >> ablate::checkError;

  //Create the RHS
  MatCreateSeqDense(PETSC_COMM_SELF, matSize, nDer, NULL, &B) >> ablate::checkError;
  PetscObjectSetName((PetscObject)B,"ablate::levelSet::RBF::rhs") >> ablate::checkError;
  MatZeroEntries(B) >> ablate::checkError;
  MatDenseGetArrayWrite(B, &vals) >> ablate::checkError;

  // Derivatives of the RBF
  for (i = 0; i < nCells; ++i) {
    for (j = 0; j < nDer; ++j) {
      vals[i + j*matSize] = Der(&x[i*dim], dx[j], dy[j], dz[j]);
    }
  }

  // Derivatives of the augmented polynomials
  if (dim == 2) {
    for (j = 0; j < nDer; ++j) {
      i = nCells;
      for (py = 0; py < p1; ++py) {
        for (px = 0; px < p1-py; ++px ){
          if(dx[j] == px && dy[j] == py) {
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
            if(dx[j] == px && dy[j] == py && dz[j] == pz) {
              vals[i + j*matSize] = fac[px]*fac[py]*fac[pz];
            }
            ++i;
          }
        }
      }
    }
  }

  MatDenseRestoreArrayWrite(B, &vals) >> ablate::checkError;

  MatViewFromOptions(B,NULL,"-ablate::levelSet::RBF::rhs_view") >> ablate::checkError;

  // Factor and solve the matrix
  MatLUFactor(A, NULL, NULL, NULL) >> ablate::checkError;
  MatMatSolve(A, B, B) >> ablate::checkError;

  MatViewFromOptions(B,NULL,"-ablate::levelSet::RBF::sol_view") >> ablate::checkError;

  // Now populate the output
  PetscReal *wt;
  PetscMalloc1(nDer*nCells, &wt) >> ablate::checkError;
  MatDenseGetArrayWrite(B, &vals) >> ablate::checkError;
  for (i = 0; i < nCells; ++i) {
    for (j = 0; j < nDer; ++j) {
      wt[i*nDer + j] = vals[i + j*matSize];
    }
  }
  MatDenseGetArrayWrite(B, &vals) >> ablate::checkError;

  *weights = wt;

  MatDestroy(&A) >> ablate::checkError;
  MatDestroy(&B) >> ablate::checkError;

}



/************ End Base Class **********************/


/************ Begin Polyharmonic Spline Derived Class **********************/
PHS::PHS(DM dm, PetscInt p, PetscInt m) : RBF(std::move(dm), std::move(p)) {
  PHS::phsOrder = m;
}

 //Polyharmonic spline: r^m
PetscReal PHS::InternalVal(PetscReal x[], PetscReal y[]) {
  PetscInt  m = PHS::phsOrder;   // The PHS order
  PetscReal r = PHS::DistanceSquared(x, y);

  return PetscPowReal(r, 0.5*((PetscReal)m));
}

// Derivatives of Polyharmonic spline at a location.
PetscReal PHS::InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) {
  PetscInt  m = PHS::phsOrder;   // The PHS order
  PetscReal r = DistanceSquared(x);

  r = PetscSqrtReal(r);

  switch (dx + 10*dy + 100*dz) {
    case 0:
      r = PetscPowReal(r, (PetscReal)m);
      break;
    case 1: // x
      r = -m*x[0]*PetscPowReal(r, (PetscReal)(m-2));
      break;
    case 2: // xx
      r = m*PetscPowReal(r, (PetscReal)(m-2)) + m*(m-2)*x[0]*x[0]*PetscPowReal(r, (PetscReal)(m-4));
      break;
    case 10: // y
      r = -m*x[1]*PetscPowReal(r, (PetscReal)(m-2));
      break;
    case 20: // yy
      r = m*PetscPowReal(r, (PetscReal)(m-2)) + m*(m-2)*x[1]*x[1]*PetscPowReal(r, (PetscReal)(m-4));
      break;
    case 100: // z
      r = -m*x[2]*PetscPowReal(r, (PetscReal)(m-2));
      break;
    case 200: // zz
      r = m*PetscPowReal(r, (PetscReal)(m-2)) + m*(m-2)*x[2]*x[2]*PetscPowReal(r, (PetscReal)(m-4));
      break;
    case 11: // xy
      r = m*(m-2)*x[0]*x[1]*PetscPowReal(r, (PetscReal)(m-4));
      break;
    case 101: // xz
      r = m*(m-2)*x[0]*x[2]*PetscPowReal(r, (PetscReal)(m-4));
      break;
    case 110: // yz
      r = m*(m-2)*x[1]*x[2]*PetscPowReal(r, (PetscReal)(m-4));
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown derivative!\n");
  }

  return r;
}
/************ End Polyharmonic Spline Derived Class **********************/


/************ Begin Multiquadric Derived Class **********************/
MQ::MQ(DM dm, PetscInt p, PetscReal scale) : RBF(std::move(dm), std::move(p)) {
  MQ::scale = scale;
}

// Multiquadric: sqrt(1+(er)^2)
PetscReal MQ::InternalVal(PetscReal x[], PetscReal y[]) {

  PetscReal h = MQ::scale;
  PetscReal e = 1.0/h;
  PetscReal r = DistanceSquared(x, y);

  return PetscSqrtReal(1.0 + e*e*r);
}

// Derivatives of Multiquadric spline at a location.
PetscReal MQ::InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) {

  PetscReal h = MQ::scale;
  PetscReal e = 1.0/h;
  PetscReal r = DistanceSquared(x);

  r = PetscSqrtReal(1.0 + e*e*r);

  switch (dx + 10*dy + 100*dz) {
    case 0:
      // Do nothing
      break;
    case 1: // x
      r = -e*e*x[0]/r;
      break;
    case 2: // xx
      r = e*e*(1.0 + e*e*(x[1]*x[1] + x[2]*x[2]))/PetscPowReal(r, 3.0);
      break;
    case 10: // y
      r = -e*e*x[1]/r;
      break;
    case 20: // yy
      r = e*e*(1.0 + e*e*(x[0]*x[0] + x[2]*x[2]))/PetscPowReal(r, 3.0);
      break;
    case 100: // z
      r = -e*e*x[2]/r;
      break;
    case 200: // zz
      r = e*e*(1.0 + e*e*(x[0]*x[0] + x[1]*x[1]))/PetscPowReal(r, 3.0);
      break;
    case 11: // xy
      r = -PetscSqr(e*e)*x[0]*x[1]/PetscPowReal(r, 3.0);
      break;
    case 101: // xz
      r = -PetscSqr(e*e)*x[0]*x[2]/PetscPowReal(r, 3.0);
      break;
    case 110: // yz
      r = -PetscSqr(e*e)*x[1]*x[2]/PetscPowReal(r, 3.0);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown derivative!\n");
  }

  return r;
}

/************ End Multiquadric Derived Class **********************/

/************ Begin Inverse Multiquadric Derived Class **********************/
IMQ::IMQ(DM dm, PetscInt p, PetscReal scale) : RBF(std::move(dm), std::move(p)) {
  IMQ::scale = scale;
}

// Multiquadric: sqrt(1+(er)^2)
PetscReal IMQ::InternalVal(PetscReal x[], PetscReal y[]) {

  PetscReal h = IMQ::scale;
  PetscReal e = 1.0/h;
  PetscReal r = DistanceSquared(x, y);

  return 1.0/PetscSqrtReal(1.0 + e*e*r);
}

// Derivatives of Multiquadric spline at a location.
PetscReal IMQ::InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) {

  PetscReal h = IMQ::scale;
  PetscReal e = 1.0/h;
  PetscReal r = DistanceSquared(x);

  r = PetscSqrtReal(1.0 + e*e*r);

  switch (dx + 10*dy + 100*dz) {
    case 0:
      r = 1.0/r;
      break;
    case 1: // x
      r = -e*e*x[0]/PetscPowReal(r, 3.0);
      break;
    case 2: // xx
      r = -e*e*(1.0 + e*e*(-2.0*x[0]*x[0] + x[1]*x[1] + x[2]*x[2]))/PetscPowReal(r, 5.0);
      break;
    case 10: // y
      r = -e*e*x[1]/PetscPowReal(r, 3.0);
      break;
    case 20: // yy
      r = -e*e*(1.0 + e*e*(x[0]*x[0] - 2.0*x[1]*x[1] + x[2]*x[2]))/PetscPowReal(r, 5.0);
      break;
    case 100: // z
      r = -e*e*x[2]/PetscPowReal(r, 3.0);
      break;
    case 200: // zz
      r = -e*e*(1.0 + e*e*(x[0]*x[0] + x[1]*x[1] - 2.0*x[2]*x[2]))/PetscPowReal(r, 5.0);
      break;
    case 11: // xy
      r = 3.0*PetscSqr(e*e)*x[0]*x[1]/PetscPowReal(r, 5.0);
      break;
    case 101: // xz
      r = 3.0*PetscSqr(e*e)*x[0]*x[2]/PetscPowReal(r, 5.0);
      break;
    case 110: // yz
      r = 3.0*PetscSqr(e*e)*x[1]*x[2]/PetscPowReal(r, 5.0);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown derivative!\n");
  }

  return r;
}
/************ End Inverse Multiquadric Derived Class **********************/

/************ Begin Gaussian Derived Class **********************/
GA::GA(DM dm, PetscInt p, PetscReal scale) : RBF(std::move(dm), std::move(p)) {
  GA::scale = scale;
}

// Gaussian: r^m
PetscReal GA::InternalVal(PetscReal x[], PetscReal y[]) {

  PetscReal h = GA::scale;
  PetscReal e2 = 1.0/(h*h);
  PetscReal r = DistanceSquared(x, y);

  return PetscExpReal(-r*e2);
}

// Derivatives of Gaussian spline at a location.
PetscReal GA::InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) {

  PetscReal h = GA::scale;
  PetscReal e2 = 1.0/(h*h);
  PetscReal r = DistanceSquared(x);

  r = PetscExpReal(-r*e2);

  switch (dx + 10*dy + 100*dz) {
    case 0:
      // Do nothing
      break;
    case 1: // x
      r *= -2.0*e2*x[0];
      break;
    case 2: // xx
      r *= 2.0*e2*(2.0*e2*x[0]*x[0]-1.0);
      break;
    case 10: // x[1]
      r *= -2.0*e2*x[1];
      break;
    case 20: // yy
      r *= 2.0*e2*(2.0*e2*x[1]*x[1]-1.0);
      break;
    case 100: // x[2]
      r *= -2.0*e2*x[2];
      break;
    case 200: // zz
      r *= 2.0*e2*(2.0*e2*x[2]*x[2]-1.0);
      break;
    case 11: // xy
      r *= 4.0*e2*e2*x[0]*x[1];
      break;
    case 101: // xz
      r *= 4.0*e2*e2*x[0]*x[2];
      break;
    case 110: // yz
      r *= 4.0*e2*e2*x[1]*x[2];
      break;
    case 111:
      r *= 8.0*e2*e2*e2*x[1]*x[2];
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown derivative!\n");
  }

  return r;
}
/************ End Gaussian Derived Class **********************/


