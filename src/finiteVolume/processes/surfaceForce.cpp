#include "domain/RBF/phs.hpp"
#include "surfaceForce.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "levelSet/levelSetUtilities.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "mathFunctions/functionFactory.hpp"

ablate::finiteVolume::processes::SurfaceForce::SurfaceForce(PetscReal sigma) : sigma(sigma) {}

// Done once at the beginning of every run
void ablate::finiteVolume::processes::SurfaceForce::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    flow.RegisterRHSFunction(ComputeSource, this);
}

// Called every time the mesh changes
void ablate::finiteVolume::processes::SurfaceForce::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {

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


}





// Smooth a sharp VOF via averaging. A issue with this is that interface given by \alpha=0.5 will move
#include "domain/reverseRange.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/constants.hpp"
static void SmoothVOF(ablate::domain::Range cellRange, DM dm, PetscScalar *dataArray, const PetscInt fID, const PetscInt nLevels) {


  const PetscInt nCellRange = cellRange.end - cellRange.start;

  // This is used to convert from a DMPlex ID to something in cellRange
  ablate::domain::ReverseRange reverseCellRange = ablate::domain::ReverseRange(cellRange);

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
  PetscReal diff = 1.0;
  while (diff > 1.e-2) {

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
    diff = -1.0;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      if (mask[c]>0) {
        PetscInt cell = cellRange.GetPoint(c);
        PetscReal *vof;
        xDMPlexPointLocalRef(dm, cell, fID, dataArray, &vof);
        diff = PetscMax(diff, PetscAbsReal(*vof - updatedVOF[c]));
        *vof = updatedVOF[c];
      }
    }
    printf("%e\n", diff);
  }
  updatedVOF += cellRange.start;
  DMGetWorkArray(dm, nCellRange, MPIU_REAL, &updatedVOF) >> ablate::utilities::PetscUtilities::checkError;

  mask += cellRange.start;
  DMRestoreWorkArray(dm, nCellRange, MPIU_INT, &mask) >> ablate::utilities::PetscUtilities::checkError;
}

static inline PetscReal SmoothDirac(PetscReal c, PetscReal c0, PetscReal t) {
  return (PetscAbsReal(c-c0) < t ? 0.5*(1.0 + cos(M_PI*(c - c0)/t))/t : 0);
}
static int cnt = 0;
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

    ablate::domain::Range cellRange;
    subDomain.GetCellRangeWithoutGhost(nullptr, cellRange);

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
    SmoothVOF(cellRange, dmData, dataArray, dataVofID, smoothWidth);

    // Range of VOF cells to consider. This should be approximatly two cells on either side if smoothWidth==4.
    // To do all of the smoothed cells use vofRange[2] = {ablate::utilities::Constants::small, 1.0 - ablate::utilities::Constants::small}
    const PetscReal vofRange[2] = {0.25, 0.75};
//    const PetscReal vofRange[2] = {ablate::utilities::Constants::small, 1.0 - ablate::utilities::Constants::small};

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      PetscInt cell = cellRange.GetPoint(c);

      const PetscReal *vofVal;
      xDMPlexPointLocalRead(dmData, cell, dataVofID, dataArray, &vofVal);

      if ( ((*vofVal) > vofRange[0]) && ((*vofVal) < vofRange[1]) ) {
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

char fname[255];
printf("\t%d\n", cnt);
sprintf(fname, "vof%d.txt", cnt++);
FILE *f1 = fopen(fname,"w");
    // Now compute the curvature, body-force, and energy
    const ablate::domain::Field &eulerField = subDomain.GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);
    DM eulerDM = subDomain.GetFieldDM(eulerField); // Get an euler-specific DM in case it's not in the same solution vector as the VOF field
    const PetscReal sigma = process->sigma; // Surface tension coefficient
    PetscScalar *fArray;
    VecGetArray(locF, &fArray) >> utilities::PetscUtilities::checkError;

    const PetscReal vofRangeAve = 0.5*(vofRange[0] + vofRange[1]);
    const PetscReal vofRangeSpread = 0.5*(vofRange[1] - vofRange[0]);
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      PetscInt cell = cellRange.GetPoint(c);

      const PetscReal *vofVal;
      xDMPlexPointLocalRead(dmData, cell, dataVofID, dataArray, &vofVal);


PetscReal loc[3];
const PetscReal *sharpVOF;
xDMPlexPointLocalRead(dmVOF, cell, vofField.id, xArray, &sharpVOF);
DMPlexComputeCellGeometryFVM(dmVOF, cell, NULL, loc, NULL) >> ablate::utilities::PetscUtilities::checkError;


//                                    1     2         3          4
fprintf(f1,"%+f\t%+f\t%+f\t%+f\t", loc[0], loc[1], *sharpVOF, *vofVal);


      if ( ((*vofVal) > vofRange[0]) && ((*vofVal) < vofRange[1]) ) {

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
        DMPlexPointLocalFieldRead(eulerDM, cell, eulerField.id, xArray, &euler) >> utilities::PetscUtilities::checkError;
        const PetscScalar density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];

        PetscScalar *eulerSource = nullptr;
        DMPlexPointLocalFieldRef(eulerDM, cell, eulerField.id, fArray, &eulerSource) >> utilities::PetscUtilities::checkError;
//                              5     6   7
fprintf(f1,"%+f\t%+f\t%+f\t", n[0], n[1], H);

        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] = 0;
        for (PetscInt d = 0; d < dim; ++d) {
            // calculate surface force and energy
            PetscReal dirac = SmoothDirac(*vofVal, vofRangeAve, vofRangeSpread);
            PetscReal surfaceForce = -dirac*(sigma * H * n[d]);
            PetscReal vel = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density;
            PetscReal surfaceEnergy = surfaceForce * vel;

            // add in the contributions
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = surfaceForce;
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += surfaceEnergy;
//                              8  9
            fprintf(f1,"%+f\t", surfaceForce);

        }

        for (PetscInt d = 0; d < dim; ++d) {
            // calculate surface force and energy
            PetscReal vel = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density;
//                              10 11
            fprintf(f1,"%+f\t", vel);

        }
      }
      else {

        fprintf(f1,"%+f\t%+f\t%+f\t", 0.0, 0.0, 0.0);


        // Zero out everything away from the interface
        PetscScalar *eulerSource = nullptr;
        DMPlexPointLocalFieldRef(eulerDM, cell, eulerField.id, fArray, &eulerSource) >> utilities::PetscUtilities::checkError;
        eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] = 0;
        for (PetscInt d = 0; d < dim; ++d) {
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = 0.0;
        }

        for (PetscInt d = 0; d < 2*dim; ++d) {
          fprintf(f1,"%+f\t", 0.0);
        }
      }
      fprintf(f1, "\n");


    }
fclose(f1);

//    PetscPrintf(MPI_COMM_WORLD, "(%s:%d, %s)\n", __FILE__, __LINE__, __FUNCTION__);
//    exit(0);


    // Cleanup
    VecRestoreArray(locF, &fArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(locX, &xArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArray(dataVec, &dataArray) >> utilities::PetscUtilities::checkError;
    DMRestoreLocalVector(dmData, &dataVec) >> utilities::PetscUtilities::checkError;

    subDomain.RestoreRange(cellRange);


    PetscFunctionReturn(PETSC_SUCCESS);
}

ablate::finiteVolume::processes::SurfaceForce::~SurfaceForce() {
  if (dmData) DMDestroy(&dmData);
}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::SurfaceForce, "calculates surface tension force and adds source terms",
         ARG(PetscReal, "sigma", "sigma, surface tension coefficient"));
