#include "domain/RBF/phs.hpp"
#include "surfaceForce.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "levelSet/levelSetUtilities.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "mathFunctions/functionFactory.hpp"

#define xexit(S, ...) {PetscFPrintf(MPI_COMM_WORLD, stderr, \
  "\x1b[1m(%s:%d, %s)\x1b[0m\n  \x1b[1m\x1b[90mexiting:\x1b[0m " S "\n", \
  __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); exit(0);}
ablate::finiteVolume::processes::SurfaceForce::SurfaceForce(PetscReal sigma) : sigma(sigma) {}

// Done once at the beginning of every run
void ablate::finiteVolume::processes::SurfaceForce::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    flow.RegisterRHSFunction(ComputeSource, this);
}

// Called every time the mesh changes
void ablate::finiteVolume::processes::SurfaceForce::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {

//  if (cellRange) subDomain.RestoreRange(cellRange);
//  if (vertRange) subDomain.RestoreRange(vertRange);
  if (dmData) DMDestroy(&dmData);

  auto& subDomain = solver.GetSubDomain();
  const PetscInt dim = subDomain.GetDimensions();
  const ablate::domain::Field& vofField = subDomain.GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
  DM dmVOF = subDomain.GetFieldDM(vofField);

  // Clone the original DM so that we can create (work) fields over it
  PetscBool isSimplex;
  DMPlexIsSimplex(dmVOF, &isSimplex) >> utilities::PetscUtilities::checkError;
  DMClone(dmVOF, &dmData) >> utilities::PetscUtilities::checkError;


  // This will create a field for the normal vector
  PetscFE fe_coords;
  PetscInt k = 1;
  PetscFECreateLagrange(PETSC_COMM_SELF, dim, dim, isSimplex, k, PETSC_DETERMINE, &fe_coords) >> utilities::PetscUtilities::checkError;
  DMSetField(dmData, dataNormalID, NULL, (PetscObject)fe_coords) >> utilities::PetscUtilities::checkError;
  PetscFEDestroy(&fe_coords) >> utilities::PetscUtilities::checkError;


  // Create a field for the smoothed VOF
  PetscFV fvm;
  PetscFVCreate(PETSC_COMM_SELF, &fvm) >> utilities::PetscUtilities::checkError;
  PetscFVSetNumComponents(fvm, 1) >> utilities::PetscUtilities::checkError;
  PetscFVSetSpatialDimension(fvm, dim) >> utilities::PetscUtilities::checkError;
  DMSetField(dmData, dataVofID, NULL, (PetscObject)fvm) >> utilities::PetscUtilities::checkError;
  PetscFVDestroy(&fvm) >> utilities::PetscUtilities::checkError;


  DMCreateDS(dmData) >> utilities::PetscUtilities::checkError;

  // Now get the regular and reverse cell range. This is stored here so that it isn't re-computed every timestep
  subDomain.GetCellRangeWithoutGhost(nullptr, cellRange);
  reverseCellRange = ablate::domain::ReverseRange(cellRange);

  subDomain.GetRange(nullptr, 0, vertRange);
  reverseVertRange = ablate::domain::ReverseRange(vertRange);


}




void SaveCellData(const char fname[255], DM dm, Vec vec, PetscInt dim, PetscInt nc, ablate::domain::Range range, const PetscInt id) {


  PetscReal    *array, *val;

  VecGetArray(vec, &array) >> ablate::utilities::PetscUtilities::checkError;

  FILE *f1 = fopen(fname, "w");

  for (PetscInt c = range.start; c < range.end; ++c) {
    PetscInt cell = range.points ? range.points[c] : c;

    PetscReal x0[3];
    DMPlexComputeCellGeometryFVM(dm, cell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;
    DMPlexPointLocalFieldRef(dm, cell, id, array, &val) >> ablate::utilities::PetscUtilities::checkError;

    for (PetscInt d = 0; d < dim; ++d) {
      fprintf(f1, "%+f\t", x0[d]);
    }
    for (PetscInt d = 0; d < nc; ++d) {
      fprintf(f1, "%+f\t", val[d]);
    }
    fprintf(f1, "\n");
  }

  fclose(f1);

  VecRestoreArray(vec, &array) >> ablate::utilities::PetscUtilities::checkError;

}


void SaveVertexData(const char fname[255], DM dm, Vec vec, PetscInt dim, PetscInt nc, ablate::domain::Range range, const PetscInt id) {

  PetscReal    *array, *val;

  VecGetArray(vec, &array) >> ablate::utilities::PetscUtilities::checkError;

  FILE *f1 = fopen(fname, "w");

  for (PetscInt v = range.start; v < range.end; ++v) {
    PetscInt vert = range.points ? range.points[v] : v;
    PetscScalar *coords;

    DMPlexPointLocalFieldRef(dm, vert, id, array, &val) >> ablate::utilities::PetscUtilities::checkError;

    DMPlexVertexGetCoordinates(dm, 1, &vert, &coords);

    for (PetscInt d = 0; d < dim; ++d) {
      fprintf(f1, "%+.16e\t", coords[d]);
    }
    for (PetscInt d = 0; d < nc; ++d) {
      fprintf(f1, "%+.16e\t", val[d]);
    }
    fprintf(f1,"\n");

    DMPlexVertexRestoreCoordinates(dm, 1, &vert, &coords);
  }

  fclose(f1);

  VecRestoreArray(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
}


#include "utilities/petscSupport.hpp"
#include "utilities/constants.hpp"
// Compute the level-set field that corresponds to a given VOF field
// The steps are:
//  1 - Determine the level-set field in cells containing a VOF value between 0 and 1
//  2 - Mark the required number of vertices (based on the cells) next to the interface cells
//  3 - Iterate over vertices EXCEPT for those with cut-cells until converged
//  4 - We may want to look at a fourth step which improve the accuracy
void Reinitialize(DM dm, Vec dataVec, const PetscInt vofID, const PetscInt gradID, const ablate::domain::Range cellRange, const ablate::domain::ReverseRange reverseCellRange, const ablate::domain::Range vertRange, const ablate::domain::ReverseRange reverseVertRange){

  const PetscInt nLevels = 4; // Number of cells surrounding cut-cells to consider

  // The data that is setup in initialize()
  PetscInt dim;

  DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

  PetscScalar *dataArray;
  VecGetArray(dataVec, &dataArray) >> ablate::utilities::PetscUtilities::checkError;


  // cellMask values are: 0: cell far from interface, do not update, 1: cut-cell, do not update, 2: cell to update
  PetscInt *cellMask, *vertMask;
  DMGetWorkArray(dm, cellRange.end - cellRange.start, MPIU_INT, &cellMask) >> ablate::utilities::PetscUtilities::checkError;
  PetscArrayzero(cellMask, cellRange.end - cellRange.start) >> ablate::utilities::PetscUtilities::checkError;
  cellMask -= cellRange.start; // This HAS to be done after the array zero call otherwise the incorrect memory location will be set to zero.

  DMGetWorkArray(dm, vertRange.end - vertRange.start, MPIU_INT, &vertMask) >> ablate::utilities::PetscUtilities::checkError;
  PetscArrayzero(vertMask, vertRange.end - vertRange.start) >> ablate::utilities::PetscUtilities::checkError;
  vertMask -= vertRange.start; // This HAS to be done after the array zero call otherwise the incorrect memory location will be set to zero.

  // Transform the VOF to an approximate level-set field via (1 - 2*vof)*h. This ensures that the inside (VOF=1) is < 0.
  // Also mark all of the cells that need updating
  PetscReal h;
  DMPlexGetMinRadius(dm, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size

  // Maximum distance along an edge.
  PetscReal gMin[3], gMax[3], maxDist = -1.0;
  DMGetBoundingBox(dm, gMin, gMax) >> ablate::utilities::PetscUtilities::checkError;
  for (PetscInt d = 0; d < dim; ++d) {
    maxDist = PetscMax(maxDist, gMax[d] - gMin[d]);
  }

  // First mark all cut-cell and get their values.
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);
    PetscReal *val;
    xDMPlexPointLocalRef(dm, cell, vofID, dataArray, &val) >> ablate::utilities::PetscUtilities::checkError;

    if ( ((*val) >= ablate::utilities::Constants::small) && ((*val) <= (1.0 - ablate::utilities::Constants::small)) ) {
      cellMask[c] = 1;                  // Mark as a cut-cell and do not update

      *val = 0.5*(1.0 - 2.0*(*val))*h;  // Convert the VOF value to an approximate level-set value at the cell-center
    }
    else {
      *val = -PetscSignReal(*val - 0.5)*maxDist;
    }

  }

  // Now get the neighboring cells, which will be updated
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    if (cellMask[c]==1) {
      PetscInt cell = cellRange.GetPoint(c);

      // Get the region of update cells
      PetscInt nc, *cellList;
      DMPlexGetNeighbors(dm, cell, nLevels, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nc, &cellList) >> ablate::utilities::PetscUtilities::checkError;

      // Level set value at the cut-cell
      const PetscReal *ls0;
      PetscReal x0[dim];
      xDMPlexPointLocalRead(dm, cell, vofID, dataArray, &ls0) >> ablate::utilities::PetscUtilities::checkError;
      DMPlexComputeCellGeometryFVM(dm, cell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;


      for (PetscInt i = 0; i < nc; ++i) {
        PetscInt id = reverseCellRange.GetIndex(cellList[i]);
        if (cellMask[id]==0) {  // Cut-cells are excluded from updating so that the interface doesn't move
          cellMask[id] = 2;       // Mark as a cell to update
          PetscReal *val;
          xDMPlexPointLocalRef(dm, cellList[i], vofID, dataArray, &val) >> ablate::utilities::PetscUtilities::checkError;
          *val = 2.0*PetscSignReal(*val)*h;

//          PetscReal x[dim];
//          DMPlexComputeCellGeometryFVM(dm, cellList[i], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;

//          PetscReal dist = 0.0;
//          for (PetscInt d = 0; d < dim; ++d) dist += PetscSqr(x[d] - x0[d]);

//          PetscReal newVal = *ls0 + PetscSignReal(*val)*PetscSqrtReal(dist);

//          if (PetscAbsReal(newVal) < PetscAbsReal(*val)) *val = newVal;

          // Now mark all of the vertices
          PetscInt nVert, *verts;
          DMPlexCellGetVertices(dm, cellList[i], &nVert, &verts) >> ablate::utilities::PetscUtilities::checkError;
          for (PetscInt j = 0; j < nVert; ++j) {
            PetscInt id = reverseVertRange.GetIndex(vertRange.GetPoint(verts[j]));
            vertMask[id] = 1;
          }
          DMPlexCellRestoreVertices(dm, cellList[i], &nVert, &verts) >> ablate::utilities::PetscUtilities::checkError;

        }
      }

      DMPlexRestoreNeighbors(dm, cell, nLevels, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nc, &cellList) >> ablate::utilities::PetscUtilities::checkError;
    }
  }

//for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//  PetscInt cell = cellRange.GetPoint(c);

//  PetscReal *val, x[3];
//  xDMPlexPointLocalRef(dm, cell, vofID, dataArray, &val) >> ablate::utilities::PetscUtilities::checkError;
//  DMPlexComputeCellGeometryFVM(dm, cell, NULL, x, NULL);
//  *val = sqrt(x[0]*x[0] + x[1]*x[1]) - 1.0;
//}


SaveCellData("ls0.txt", dm, dataVec, dim, 1, cellRange, vofID);
xexit("");

  PetscReal diff = 1.0, dt = 0.1*h;
  PetscInt iter = 0;

  PetscReal *cellNormal;
  DMGetWorkArray(dm, dim*(cellRange.end - cellRange.start), MPIU_REAL, &cellNormal) >> ablate::utilities::PetscUtilities::checkError;
  cellNormal -= dim*cellRange.start;

  while (diff > 1.e-5) {
    ++iter;

    // Get the gradients at the vertices
    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
      if (vertMask[v]==1) {
        PetscInt vert = vertRange.GetPoint(v);
        PetscReal *g;

        xDMPlexPointLocalRef(dm, vert, gradID, dataArray, &g) >> ablate::utilities::PetscUtilities::checkError;

        DMPlexVertexGradFromCell(dm, vert, dataVec, vofID, 0, g);

      }
    }

if (iter%100==0) {
  char fname1[255];
  sprintf(fname1, "grad%d.txt", iter);
  SaveVertexData(fname1, dm, dataVec, dim, dim, vertRange, gradID);
}
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      if (cellMask[c]==2) {
        PetscInt cell = cellRange.GetPoint(c);
        DMPlexCellGradFromCell(dm, cell, dataVec, vofID, 0, &cellNormal[c*dim]);
      }
    }


    diff = -1.0;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      if (cellMask[c]==2) {
        PetscInt cell = cellRange.GetPoint(c);

        PetscInt nVert, *verts;
        PetscReal g[3] = {0.0, 0.0, 0.0}, totalWts = 0.0;
        PetscReal *vertLocs;
        PetscReal x0[dim];

        DMPlexComputeCellGeometryFVM(dm, cell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexCellGetVertices(dm, cell, &nVert, &verts) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexVertexGetCoordinates(dm, nVert, verts, &vertLocs) >> ablate::utilities::PetscUtilities::checkError;

        // Get the normal at the cell center using averaging
        PetscReal *n = &cellNormal[c*dim];

//        PetscReal n[3] = {0.0, 0.0, 0.0};
//        for (PetscInt v = 0; v < nVert; ++v) {

//          PetscReal *vertGrad;
//          xDMPlexPointLocalRef(dm, verts[v], gradID, dataArray, &vertGrad) >> ablate::utilities::PetscUtilities::checkError;

//          PetscReal mag = ablate::utilities::MathUtilities::MagVector(dim, vertGrad);
//          if (PetscAbsReal(mag) > ablate::utilities::Constants::tiny) {
//            for (PetscInt d = 0; d < dim; ++d){
//              vertGrad[d] /= mag;
//              n[d] += vertGrad[d];
//            }
//          }
//        }
//        for (PetscInt d = 0; d < dim; ++d) n[d] /= nVert;

        for (PetscInt v = 0; v < nVert; ++v) {

          PetscReal *vertGrad;
          xDMPlexPointLocalRef(dm, verts[v], gradID, dataArray, &vertGrad) >> ablate::utilities::PetscUtilities::checkError;

          PetscReal direction[dim];
          for (PetscInt d = 0; d < dim; ++d) {
            direction[d] = x0[d] - vertLocs[v*dim + d];
          }
          ablate::utilities::MathUtilities::NormVector(dim, direction);

          PetscReal wts = ablate::utilities::MathUtilities::DotVector(dim, n, vertGrad);

          wts = PetscMax(0.0, wts);

          totalWts += wts;
          for (PetscInt d = 0; d < dim; ++d) {
            g[d] += wts*vertGrad[d];
          }

        }
        DMPlexVertexRestoreCoordinates(dm, nVert, verts, &vertLocs) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexCellRestoreVertices(dm, cell, &nVert, &verts) >> ablate::utilities::PetscUtilities::checkError;

//        if (totalWts<ablate::utilities::Constants::small) {
//          throw std::runtime_error("Could not find upwind direction for cell " + std::to_string(cell) + ".");
//        }

//printf("%+e\n", totalWts);
//xexit("");

        // In cases where the level-set values are flat don't normalize
        if (totalWts > ablate::utilities::Constants::tiny) {
          for (PetscInt d = 0; d < dim; ++d) {
            g[d] /= totalWts;
          }
        }


        PetscReal *phi;
        xDMPlexPointLocalRef(dm, cell, vofID, dataArray, &phi) >> ablate::utilities::PetscUtilities::checkError;

        PetscReal SG = (*phi)/PetscSqrtReal(PetscSqr(*phi) + PetscSqr(h));  // Sign function
        SG *= dt*(ablate::utilities::MathUtilities::MagVector(dim, g) - 1.0); // dt*sgn(\phi)*(\|\nabla \phi \| - 1.0)

        diff = PetscMax(diff, PetscAbsReal(SG));
        *phi -= SG;

      }
    }
if (iter%100==0) {
  char fname[255];
  sprintf(fname, "ls%d.txt", iter);
  SaveCellData(fname, dm, dataVec, dim, 1, cellRange, vofID);
}
    printf("%4d: %e\n", iter, diff);
//xexit("");
  }
  xexit("");

  cellNormal += dim*cellRange.start;
  DMRestoreWorkArray(dm, dim*(cellRange.end - cellRange.start), MPIU_REAL, &cellNormal) >> ablate::utilities::PetscUtilities::checkError;


  vertMask += vertRange.start;
  DMRestoreWorkArray(dm, vertRange.end - vertRange.start, MPIU_INT, &vertMask) >> ablate::utilities::PetscUtilities::checkError;

  cellMask += cellRange.start;
  DMRestoreWorkArray(dm, cellRange.end - cellRange.start, MPIU_INT, &cellMask) >> ablate::utilities::PetscUtilities::checkError;

}


// Smooth a sharp VOF via averaging. A issue with this is that interface given by \alpha=0.5 will move
void SmoothVOF(ablate::domain::Range cellRange, ablate::domain::ReverseRange reverseCellRange, DM dm, Vec dataVec, const PetscInt fID, const PetscInt nLevels) {


  const PetscInt nCellRange = cellRange.end - cellRange.start;


  PetscScalar *dataArray;
  VecGetArray(dataVec, &dataArray);

  // A mask indicating the cut-cells. 0: Do not update, 1: A cut-cell (do not change), 2: Update
  PetscInt *mask;
  DMGetWorkArray(dm, nCellRange, MPIU_INT, &mask) >> ablate::utilities::PetscUtilities::checkError;
  PetscArrayzero(mask, nCellRange) >> ablate::utilities::PetscUtilities::checkError;
  mask -= cellRange.start; // This HAS to be done after the array zero call otherwise the incorrect memory location will be set to zero.

  // Mark all cut-cells as level-1
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);

    const PetscReal *vof;
    xDMPlexPointLocalRead(dm, cell, fID, dataArray, &vof);

    if ( ((*vof) > ablate::utilities::Constants::small) && ((*vof) < (1.0 - ablate::utilities::Constants::small)) ) {
      mask[c] = 1;
    }
  }

  // Mark the rest of the points
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    if (mask[c]==1) {
      PetscInt cell = cellRange.GetPoint(c);
      PetscInt nCells, *cells;
      DMPlexGetNeighbors(dm, cell, nLevels, -1.0, -1.0, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

      for (PetscInt i = 0; i < nCells; ++i) {
        PetscInt rc = reverseCellRange.GetIndex(cells[i]);
        if (mask[rc]==0) mask[rc] = 2; // DMPlexGetNeighbors also returns the center cell, so check if the cell has already been marked
      }

      DMPlexRestoreNeighbors(dm, nLevels, nLevels, -1.0, -1.0, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

    }
  }


  // This will have an issue on parallel runs -- The information will need to get communicated between processors via ghost-cells
  PetscReal *updatedVOF;
  DMGetWorkArray(dm, nCellRange, MPIU_REAL, &updatedVOF) >> ablate::utilities::PetscUtilities::checkError;
  updatedVOF -= cellRange.start;
//  PetscReal diff = 1.0;
//  while (diff > 1.e-1) {
  PetscInt iter = 0;
  while ( (iter++) < 4) {

    // Update all marked cells as the average of the surrounding nearest-neighbors cells
    // Should this store the stencil so as not to redo DMPlexGetNeighbors multiple times?
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      if (mask[c]>0) {
        PetscInt cell = cellRange.GetPoint(c);
        PetscInt nCells, *cells;
        DMPlexGetNeighbors(dm, cell, 1, -1.0, -1.0, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

        updatedVOF[c] = 0.0;
        for (PetscInt i = 0; i < nCells; ++i) {
          const PetscReal *vof;
          xDMPlexPointLocalRead(dm, cells[i], fID, dataArray, &vof);
          updatedVOF[c] += *vof;
        }
        updatedVOF[c] /= nCells;

        DMPlexRestoreNeighbors(dm, cell, 1, -1.0, -1.0, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
      }
    }

    // Copy over the data and check for convergence
//    diff = -1.0;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      if (mask[c]>0) {
        PetscInt cell = cellRange.GetPoint(c);
        PetscReal *vof;
        xDMPlexPointLocalRef(dm, cell, fID, dataArray, &vof);
//        diff = PetscMax(diff, PetscAbsReal(*vof - updatedVOF[c]));
        *vof = updatedVOF[c];
      }
    }
//    printf("%e\n", diff);
  }

  VecRestoreArray(dataVec, &dataArray);

  updatedVOF += cellRange.start;
  DMGetWorkArray(dm, nCellRange, MPIU_REAL, &updatedVOF) >> ablate::utilities::PetscUtilities::checkError;

  mask += cellRange.start;
  DMRestoreWorkArray(dm, nCellRange, MPIU_INT, &mask) >> ablate::utilities::PetscUtilities::checkError;
}



void ExtendVOF(ablate::domain::Range cellRange, ablate::domain::ReverseRange reverseCellRange, DM dm, Vec dataVec, const PetscInt fID, const PetscInt nLevels) {


  const PetscInt nCellRange = cellRange.end - cellRange.start;

  PetscScalar *dataArray;
  VecGetArray(dataVec, &dataArray);

  // A mask indicating the cut-cells. 0: Do not update, 1: A cut-cell (do not change), 2: Update
  PetscInt *mask, *count, *vof0;
  DMGetWorkArray(dm, nCellRange, MPIU_INT, &mask) >> ablate::utilities::PetscUtilities::checkError;
  PetscArrayzero(mask, nCellRange) >> ablate::utilities::PetscUtilities::checkError;
  mask -= cellRange.start; // This HAS to be done after the array zero call otherwise the incorrect memory location will be set to zero.

  DMGetWorkArray(dm, nCellRange, MPIU_INT, &count) >> ablate::utilities::PetscUtilities::checkError;
  PetscArrayzero(count, nCellRange) >> ablate::utilities::PetscUtilities::checkError;
  count -= cellRange.start; // This HAS to be done after the array zero call otherwise the incorrect

  DMGetWorkArray(dm, nCellRange, MPIU_INT, &vof0) >> ablate::utilities::PetscUtilities::checkError;
  PetscArrayzero(vof0, nCellRange) >> ablate::utilities::PetscUtilities::checkError;
  vof0 -= cellRange.start; // This HAS to be done after the array zero call otherwise the incorrect

  // Mark all cut-cells as level-1
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);

    PetscReal *vof;
    xDMPlexPointLocalRef(dm, cell, fID, dataArray, &vof);

    if ( ((*vof) > ablate::utilities::Constants::small) && ((*vof) < (1.0 - ablate::utilities::Constants::small)) ) {
      mask[c] = 1;
    }
    else {
      vof0[c] = (*vof > 0.5) ? +1 : 0;
      *vof = 1.0; // To start the multiplication
    }
  }

  // Mark the remaining levels
  for (PetscInt currentLevel = 2; currentLevel < nLevels+1; ++currentLevel) {
    const PetscInt prevLevel = currentLevel - 1;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      if (mask[c] == prevLevel) {
        PetscInt cell = cellRange.GetPoint(c);
        PetscInt nCells, *cells;
        const PetscReal *vof;
        xDMPlexPointLocalRead(dm, cell, fID, dataArray, &vof);
        DMPlexGetNeighbors(dm, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

        for (PetscInt n = 0; n < nCells; ++n){
          PetscInt id = reverseCellRange.GetIndex(cells[n]);
          if (mask[id]==0 || mask[id]==currentLevel){
            mask[id] = currentLevel;
            ++count[id];

            PetscReal *val;
            xDMPlexPointLocalRef(dm, cells[n], fID, dataArray, &val);

            // This is equivalent to sign[id]*vof[0] + (1-sign[id])*(1.0-vof[0]);
            *val *= (vof0[id] == +1) ? *vof : 1.0 - *vof;

          }
        }
        DMPlexRestoreNeighbors(dm, cell, 1, -1.0, -1, PETSC_TRUE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
      }
    }

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      if (mask[c] == currentLevel) {
        PetscInt cell = cellRange.GetPoint(c);
        PetscReal *val;
        xDMPlexPointLocalRef(dm, cell, fID, dataArray, &val);
        *val = PetscPowReal(*val, 1.0/count[c]);

        // This is equivalent to sign[c]*(1.0 + val[0]) + (1-sign[c])*(-val[0]);
        *val = (vof0[c] == +1) ? 1.0 + *val : -*val;
      }
    }

  }

  // Set everything else
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);
    PetscReal *val;
    xDMPlexPointLocalRef(dm, cell, fID, dataArray, &val);
    if (mask[c] == 0) {
      *val = NAN;
    }
  }

  VecRestoreArray(dataVec, &dataArray);

  vof0 += cellRange.start;
  DMRestoreWorkArray(dm, nCellRange, MPIU_INT, &vof0) >> ablate::utilities::PetscUtilities::checkError;

  count += cellRange.start;
  DMRestoreWorkArray(dm, nCellRange, MPIU_INT, &count) >> ablate::utilities::PetscUtilities::checkError;

  mask += cellRange.start;
  DMRestoreWorkArray(dm, nCellRange, MPIU_INT, &mask) >> ablate::utilities::PetscUtilities::checkError;





}

inline PetscReal SmoothDirac(PetscReal c, PetscReal c0, PetscReal t) {
  return (PetscAbsReal(c-c0) < t ? 0.5*(1.0 + cos(M_PI*(c - c0)/t))/t : 0);
}
static int cnt = -1;
PetscErrorCode ablate::finiteVolume::processes::SurfaceForce::ComputeSource(const FiniteVolumeSolver &flow, DM dm, PetscReal time, Vec locX, Vec locF, void *ctx) {
    PetscFunctionBegin;

    ablate::finiteVolume::processes::SurfaceForce *process = (ablate::finiteVolume::processes::SurfaceForce *)ctx;
    const ablate::domain::SubDomain& subDomain = flow.GetSubDomain();
    const PetscInt dim = subDomain.GetDimensions();

    // Look for the euler field and volume fraction (alpha)
    const ablate::domain::Field& vofField = subDomain.GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);

    // Get the DMs of the two fields. Each is obtained in case they change from SOL to AUX at some point in the future.
    // Note that the physical layout of the two DMs will be the same, but which DM the VECs are stored in
    DM dmVOF = subDomain.GetFieldDM(vofField);


    // The data that is setup in initialize()
    DM dmData = process->dmData;
    const PetscInt dataVofID = process->dataVofID, dataNormalID = process->dataNormalID;
    ablate::domain::Range cellRange = process->cellRange;
    ablate::domain::ReverseRange reverseCellRange = process->reverseCellRange;

    Vec dataVec;
    DMGetLocalVector(dmData, &dataVec) >> utilities::PetscUtilities::checkError;
    VecSet(dataVec, NAN); // Set it to NaN to know when stuff has been set (or not).

    PetscScalar *dataArray;
    VecGetArray(dataVec, &dataArray) >> utilities::PetscUtilities::checkError;

    const PetscScalar *xArray;
    VecGetArrayRead(locX, &xArray) >> utilities::PetscUtilities::checkError;


    // Copy the sharp VOF so that it can be smoothed
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      PetscInt cell = cellRange.GetPoint(c);
      const PetscReal *xVOF;
      PetscReal *dVOF;

      xDMPlexPointLocalRead(dmVOF, cell, vofField.id, xArray, &xVOF);
      xDMPlexPointLocalRef(dmData, cell, dataVofID, dataArray, &dVOF);

      *dVOF = *xVOF;

    }

    // A width of 4 on either side of the interface seems to work well
    const PetscInt smoothWidth = 4;
    ExtendVOF(cellRange, reverseCellRange, dmData, dataVec, dataVofID, smoothWidth+2);
    SmoothVOF(cellRange, reverseCellRange, dmData, dataVec, dataVofID, smoothWidth);


    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      PetscInt cell = cellRange.GetPoint(c);

      const PetscReal *vofVal;
      xDMPlexPointLocalRead(dmData, cell, dataVofID, dataArray, &vofVal);

      if (!PetscIsNanReal(*vofVal)) {
        PetscInt nVerts, *verts;
        DMPlexCellGetVertices(dmVOF, cell, &nVerts, &verts);

        for (PetscInt v = 0; v < nVerts; ++v) {
          PetscInt vert = verts[v];

          PetscReal *n;
          xDMPlexPointLocalRef(dmData, vert, dataNormalID, dataArray, &n);

          if (isnan(n[0])) {
            DMPlexVertexGradFromCell(dmData, vert, dataVec, dataVofID, 0, n);
            PetscReal mag = ablate::utilities::MathUtilities::MagVector(dim, n);
            for (PetscInt d = 0; d < dim; ++d) n[d] /= -mag;
          }
        }
        DMPlexCellRestoreVertices(dmVOF, cell, &nVerts, &verts);
      }
    }


    FILE *f1;
    const PetscInt step = 10;
    printf("\t%d\n", ++cnt);
    bool saveFile = (cnt%step==0);

    if (saveFile) {
      char fname[255];
      sprintf(fname, "vof%d.txt", cnt);
      f1 = fopen(fname,"w");
    }


    // Now compute the curvature, body-force, and energy
    const ablate::domain::Field &eulerField = subDomain.GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);
    DM eulerDM = subDomain.GetFieldDM(eulerField); // Get an euler-specific DM in case it's not in the same solution vector as the VOF field
    const PetscReal sigma = process->sigma; // Surface tension coefficient
    PetscScalar *fArray;
    VecGetArray(locF, &fArray) >> utilities::PetscUtilities::checkError;

    // The grid spacing
    PetscReal h;
    DMPlexGetMinRadius(dm, &h) >> ablate::utilities::PetscUtilities::checkError;
    h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      PetscInt cell = cellRange.GetPoint(c);

      const PetscReal *vofVal;
      xDMPlexPointLocalRead(dmData, cell, dataVofID, dataArray, &vofVal);

      // Estimation of the signed-distance function
      const PetscReal phi = 0.5*(1.0 - 2*vofVal[0])*h;
      const PetscReal dirac = SmoothDirac(phi, 0.0, 2.0*h);


//      const PetscReal dirac = 1.0;



      const PetscReal *sharpVOF;
      xDMPlexPointLocalRead(dmVOF, cell, vofField.id, xArray, &sharpVOF);
      if (saveFile){
        PetscReal loc[3];
        DMPlexComputeCellGeometryFVM(dmVOF, cell, NULL, loc, NULL) >> ablate::utilities::PetscUtilities::checkError;
        //                                    1     2         3          4
        fprintf(f1,"%+f\t%+f\t%+f\t%+f\t", loc[0], loc[1], *sharpVOF, *vofVal);
      }


      if (dirac > 0.0) {
//      if ( ((*sharpVOF) > ablate::utilities::Constants::small) && ((*sharpVOF) < (1.0 - ablate::utilities::Constants::small)) ) {


        PetscReal n[3] = {0.0, 0.0, 0.0}; // Normal at the cell-center

        PetscInt nVerts, *verts;
        DMPlexCellGetVertices(dmVOF, cell, &nVerts, &verts);
        for (PetscInt v = 0; v < nVerts; ++v) {
          PetscInt vert = verts[v];

          const PetscReal *vertNormal;
          xDMPlexPointLocalRead(dmData, vert, dataNormalID, dataArray, &vertNormal);

          // The normal at the cell-center will be the average of the vertices
          for (PetscInt d = 0; d < dim; ++d) {
            n[d] += vertNormal[d];
          }
        }
        for (PetscInt d = 0; d < dim; ++d) n[d] /= nVerts;
        DMPlexCellRestoreVertices(dmVOF, cell, &nVerts, &verts);

        // Divergence of the normal to give the total curvature
        PetscReal H = 0.0;
        for (PetscInt d = 0; d < dim; ++d) {
          PetscReal g[dim];
          DMPlexCellGradFromVertex(dmData, cell, dataVec, dataNormalID, d, g);
          H += g[d];
        }

        const PetscScalar *euler = nullptr;
        xDMPlexPointLocalRead(eulerDM, cell, eulerField.id, xArray, &euler) >> utilities::PetscUtilities::checkError;
        const PetscScalar density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];

        PetscScalar *eulerSource = nullptr;
        xDMPlexPointLocalRef(eulerDM, cell, eulerField.id, fArray, &eulerSource) >> utilities::PetscUtilities::checkError;
//                                                      5     6   7
        if (saveFile) fprintf(f1,"%+f\t%+f\t%+f\t", n[0], n[1], H);

        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] = 0;
        for (PetscInt d = 0; d < dim; ++d) {
            // calculate surface force and energy
            PetscReal surfaceForce = -dirac* sigma * H * n[d];
            PetscReal vel = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density;
            PetscReal surfaceEnergy = surfaceForce * vel;

            // add in the contributions
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = surfaceForce;
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += surfaceEnergy;
//                                                  8  9
            if (saveFile) fprintf(f1,"%+f\t", surfaceForce);

        }


        if (saveFile) {
          for (PetscInt d = 0; d < dim; ++d) {
            PetscReal vel = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density;
            //                10 11
            fprintf(f1,"%+f\t", vel);

          }

          const ablate::domain::Field *pressureField = &(subDomain.GetField(ablate::finiteVolume::CompressibleFlowFields::PRESSURE_FIELD));
          Vec pressureVec = subDomain.GetVec(*pressureField);
          DM pressureDM = subDomain.GetFieldDM(*pressureField);
          const PetscReal *pressureArray, *pressureVal;
          VecGetArrayRead(pressureVec, &pressureArray);
          xDMPlexPointLocalRead(pressureDM, cell, dataVofID, dataArray, &pressureVal);
          fprintf(f1, "%+f\t", *pressureVal);
          VecRestoreArrayRead(pressureVec, &pressureArray);


        }
      }
      else {

        if (saveFile) fprintf(f1,"%+f\t%+f\t%+f\t", 0.0, 0.0, 0.0);


        // Zero out everything away from the interface
        PetscScalar *eulerSource = nullptr;
        DMPlexPointLocalFieldRef(eulerDM, cell, eulerField.id, fArray, &eulerSource) >> utilities::PetscUtilities::checkError;
        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] = 0;
        for (PetscInt d = 0; d < dim; ++d) {
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = 0.0;
        }


        if (saveFile) {
          for (PetscInt d = 0; d < 2*dim; ++d) {
            fprintf(f1,"%+f\t", 0.0);
          }
          fprintf(f1, "%+f\t", 0.0);
        }


      }


      if (saveFile) fprintf(f1, "\n");


    }
    if (saveFile) fclose(f1);


    // Cleanup
    VecRestoreArray(locF, &fArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(locX, &xArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArray(dataVec, &dataArray) >> utilities::PetscUtilities::checkError;
    DMRestoreLocalVector(dmData, &dataVec) >> utilities::PetscUtilities::checkError;

//xexit("");

    PetscFunctionReturn(PETSC_SUCCESS);
}

ablate::finiteVolume::processes::SurfaceForce::~SurfaceForce() {
  if (dmData) DMDestroy(&dmData);
}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::SurfaceForce, "calculates surface tension force and adds source terms",
         ARG(PetscReal, "sigma", "sigma, surface tension coefficient"));
