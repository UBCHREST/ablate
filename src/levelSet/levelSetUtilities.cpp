#include "levelSetUtilities.hpp"
#include <petsc.h>
#include <memory>
#include "LS-VOF.hpp"
#include "cellGrad.hpp"
#include "geometry.hpp"
#include "domain/range.hpp"
#include "domain/reverseRange.hpp"
#include "mathFunctions/functionWrapper.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"

void ablate::levelSet::Utilities::CellValGrad(DM dm, const PetscInt p, PetscReal *c, PetscReal *c0, PetscReal *g) {
    DMPolytopeType ct;
    PetscInt Nc;
    PetscReal *coords = NULL;
    const PetscScalar *array;
    PetscBool isDG;
    PetscReal x0[3];

    // Coordinates of the cell vertices
    DMPlexGetCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;

    // Center of the cell
    DMPlexComputeCellGeometryFVM(dm, p, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

    // Get the cell type and call appropriate VOF function
    DMPlexGetCellType(dm, p, &ct) >> ablate::utilities::PetscUtilities::checkError;
    switch (ct) {
        case DM_POLYTOPE_SEGMENT:
            Grad_1D(x0, coords, c, c0, g) >> ablate::utilities::PetscUtilities::checkError;
            break;
        case DM_POLYTOPE_TRIANGLE:
            Grad_2D_Tri(x0, coords, c, c0, g) >> ablate::utilities::PetscUtilities::checkError;
            break;
        case DM_POLYTOPE_QUADRILATERAL:
            Grad_2D_Quad(x0, coords, c, c0, g) >> ablate::utilities::PetscUtilities::checkError;
            break;
        case DM_POLYTOPE_TETRAHEDRON:
            Grad_3D_Tetra(x0, coords, c, c0, g) >> ablate::utilities::PetscUtilities::checkError;
            break;
        case DM_POLYTOPE_HEXAHEDRON:
            Grad_3D_Hex(x0, coords, c, c0, g) >> ablate::utilities::PetscUtilities::checkError;
            break;
        default:
            throw std::invalid_argument("No element geometry for cell " + std::to_string(p) + " with type " + DMPolytopeTypes[ct]);
    }

    DMPlexRestoreCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;
}

void ablate::levelSet::Utilities::CellValGrad(DM dm, const PetscInt fid, const PetscInt p, Vec f, PetscReal *c0, PetscReal *g) {
    PetscInt nv, *verts;
    const PetscScalar *fvals, *v;
    PetscScalar *c;

    DMPlexCellGetVertices(dm, p, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

    DMGetWorkArray(dm, nv, MPIU_SCALAR, &c) >> ablate::utilities::PetscUtilities::checkError;

    VecGetArrayRead(f, &fvals) >> utilities::PetscUtilities::checkError;

    for (PetscInt i = 0; i < nv; ++i) {
        // DMPlexPointLocalFieldRead isn't behaving like I would expect. If I don't make f a pointer then it just returns zero.
        //    Additionally, it looks like it allows for the editing of the value.
        if (fid >= 0) {
            DMPlexPointLocalFieldRead(dm, verts[i], fid, fvals, &v) >> utilities::PetscUtilities::checkError;
        } else {
            DMPlexPointLocalRead(dm, verts[i], fvals, &v) >> utilities::PetscUtilities::checkError;
        }

        c[i] = *v;
    }

    DMPlexCellRestoreVertices(dm, p, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

    ablate::levelSet::Utilities::CellValGrad(dm, p, c, c0, g);

    DMRestoreWorkArray(dm, nv, MPIU_SCALAR, &c) >> ablate::utilities::PetscUtilities::checkError;
}

void ablate::levelSet::Utilities::CellValGrad(std::shared_ptr<ablate::domain::SubDomain> subDomain, const ablate::domain::Field *field, const PetscInt p, PetscReal *c0, PetscReal *g) {
    DM dm = subDomain->GetFieldDM(*field);
    Vec f = subDomain->GetVec(*field);
    ablate::levelSet::Utilities::CellValGrad(dm, field->id, p, f, c0, g);
}

void ablate::levelSet::Utilities::VertexToVertexGrad(std::shared_ptr<ablate::domain::SubDomain> subDomain, const ablate::domain::Field *field, const PetscInt p, PetscReal *g) {
    // Given a field determine the gradient at a vertex

    DM dm = subDomain->GetFieldDM(*field);
    Vec vec = subDomain->GetVec(*field);

    DMPlexVertexGradFromVertex(dm, p, vec, field->id, 0, g) >> ablate::utilities::PetscUtilities::checkError;
}

/**
  * Compute the upwind derivative
  * @param dm - Domain of the gradient data.
  * @param gradArray - Array containing the cell-centered gradient
  * @param v - Vertex id
  * @param direction - The direction to be considered upwind. +1 for standard upwind, -1 of downwind
  * @param g - On input the gradient of the level-set field at a vertex. On output the upwind gradient at v
  */
static void VertexUpwindGrad(DM dm, PetscScalar *gradArray, const PetscInt gradID, const PetscInt v, const PetscReal direction, PetscReal *g) {
  // The upwind direction is determined using the dot product between the vector u and the vector connecting the cell-center
  //    and the vertex

  PetscInt          dim;
  PetscReal         weightTotal = 0.0;
  PetscScalar       x0[3], n[3];

  DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

  ablate::utilities::MathUtilities::NormVector(dim, g, n);

  DMPlexComputeCellGeometryFVM(dm, v, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt d = 0; d < dim; ++d) {
    g[d] = 0.0;
  }


  // Obtain all cells which use this vertex
  PetscInt nCells, *cells;
  DMPlexVertexGetCells(dm, v, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt c = 0; c < nCells; ++c) {
    PetscReal x[3];
    DMPlexComputeCellGeometryFVM(dm, cells[c], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;

    PetscReal dot = 0.0;
    for (PetscInt d = 0; d < dim; ++d) {
      dot += n[d]*(x0[d] - x[d]);
    }

    dot *= direction;

    if (dot>=0.0) {

      weightTotal += dot;

      const PetscScalar *cellGrad = nullptr;
      xDMPlexPointLocalRead(dm, cells[c], gradID, gradArray, &cellGrad) >> ablate::utilities::PetscUtilities::checkError;

      // Weighted average of the surrounding cell-center gradients.
      //  Note that technically this is (in 2D) the area of the quadrilateral that is formed by connecting
      //  the vertex, center of the neighboring edges, and the center of the triangle. As the three quadrilaterals
      //  that are formed this way all have the same area, there is no need to take into account the 1/3. Something
      //  similar should hold in 3D and for other cell types that ABLATE uses.
      for (PetscInt d = 0; d < dim; ++d) {
        g[d] += dot*cellGrad[d];
      }
    }
  }

  DMPlexVertexRestoreCells(dm, v, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

  // Size of the communicator
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);
  int size;
  MPI_Comm_size(comm, &size) >> ablate::utilities::MpiUtilities::checkError;

  // Error checking
  if ( PetscAbs(weightTotal) < ablate::utilities::Constants::small ) {
    // When running on a single processor all vertices should have an upwind cell. Throw an error if that's not the case.
    // When running in parallel, ghost vertices at the edge of the local domain may not have any surrounding upwind cells, so
    //  ignore the error and simply set the upwind gradient to zero.
    if ( size==1 ) {
      throw std::runtime_error("ablate::levelSet::Utilities::VertexUpwindGrad encounted a situation where there are no upwind cells");
    }
    for (PetscInt d = 0; d < dim; ++d) {
      g[d] = 0.0;
    }
  }
  else {
    for (PetscInt d = 0; d < dim; ++d) {
      g[d] /= weightTotal;
    }
  }
}

// Given a level set and normal at the cell center compute the level set values at the vertices assuming a straight interface
void ablate::levelSet::Utilities::VertexLevelSet_LS(DM dm, const PetscInt p, const PetscReal c0, const PetscReal *n, PetscReal **c) {
    PetscInt dim, Nc, nVerts, i, j;
    PetscReal x0[3] = {0.0, 0.0, 0.0};
    PetscReal *coords = NULL;
    const PetscScalar *array;
    PetscBool isDG;

    DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

    // The cell center
    DMPlexComputeCellGeometryFVM(dm, p, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

    // Coordinates of the cell vertices
    DMPlexGetCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;

    // Number of vertices
    nVerts = Nc / dim;

    if (*c == NULL) {
        PetscMalloc1(nVerts, c) >> ablate::utilities::PetscUtilities::checkError;
    }

    // The level set value of each vertex. This assumes that the interface is a line/plane
    //    with the given unit normal.
    for (i = 0; i < nVerts; ++i) {
        (*c)[i] = c0;
        for (j = 0; j < dim; ++j) {
            (*c)[i] += n[j] * (coords[i * dim + j] - x0[j]);
        }
    }

    DMPlexRestoreCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;



}

// Given a cell VOF and normal at the cell center compute the level set values at the vertices assuming a straight interface
void ablate::levelSet::Utilities::VertexLevelSet_VOF(DM dm, const PetscInt p, const PetscReal targetVOF, const PetscReal *n, PetscReal **c) {
    PetscReal vof;         // current VOF of the cell
    PetscReal area;        // length (2D) or area (3D) of the cell face
    PetscReal cellVolume;  // Area (2D) or volume (3D) of the cell
    const PetscReal tol = 1e-8;
    PetscInt i;
    PetscReal offset;
    PetscReal vofError;
    PetscInt nv;

    // Get the number of vertices for the cell
    DMPlexCellGetNumVertices(dm, p, &nv) >> ablate::utilities::PetscUtilities::checkError;

    // Get an initial guess at the vertex level set values assuming that the interface passes through the cell-center.
    // Also allocates c if c==NULL on entry
    ablate::levelSet::Utilities::VertexLevelSet_LS(dm, p, 0.0, n, c);

    // Get the resulting VOF from the initial guess
    ablate::levelSet::Utilities::VOF(dm, p, *c, &vof, &area, &cellVolume);
    vofError = targetVOF - vof;

    while (fabs(vofError) > tol) {
        // The amount the center level set value needs to shift by.
        offset = vofError * cellVolume / area;

        // If this isn't damped then it will overshoot and there will be no interface in the cell
        offset *= 0.5;

        for (i = 0; i < nv; ++i) {
            (*c)[i] -= offset;
        }

        ablate::levelSet::Utilities::VOF(dm, p, *c, &vof, &area, NULL);
        vofError = targetVOF - vof;
    };
}

// Returns the VOF for a given cell using the level-set values at the cell vertices.
// Refer to "Quadrature rules for triangular and tetrahedral elements with generalized functions"
//  by Holdych, Noble, and Secor, Int. J. Numer. Meth. Engng 2008; 73:1310-1327.
void ablate::levelSet::Utilities::VOF(DM dm, const PetscInt p, PetscReal *c, PetscReal *vof, PetscReal *area, PetscReal *vol) {
    DMPolytopeType ct;
    PetscInt Nc;
    PetscReal *coords = NULL;
    const PetscScalar *array;
    PetscBool isDG;

    // Coordinates of the cell vertices
    DMPlexGetCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;

    // Get the cell type and call appropriate VOF function
    DMPlexGetCellType(dm, p, &ct) >> ablate::utilities::PetscUtilities::checkError;
    switch (ct) {
        case DM_POLYTOPE_SEGMENT:
            VOF_1D(coords, c, vof, area, vol);
            break;
        case DM_POLYTOPE_TRIANGLE:
            VOF_2D_Tri(coords, c, vof, area, vol);
            break;
        case DM_POLYTOPE_QUADRILATERAL:
            VOF_2D_Quad(coords, c, vof, area, vol);
            break;
        case DM_POLYTOPE_TETRAHEDRON:
            VOF_3D_Tetra(coords, c, vof, area, vol);
            break;
        case DM_POLYTOPE_HEXAHEDRON:
            VOF_3D_Hex(coords, c, vof, area, vol);
            break;
        default:
            throw std::invalid_argument("No element geometry for cell " + std::to_string(p) + " with type " + DMPolytopeTypes[ct]);
    }

    DMPlexRestoreCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;
}

// Returns the VOF for a given cell with a known level set value (c0) and normal (nIn).
//  This computes the level-set values at the vertices by approximating the interface as a straight-line with the same normal
//  as provided
void ablate::levelSet::Utilities::VOF(DM dm, const PetscInt p, const PetscReal c0, const PetscReal *nIn, PetscReal *vof, PetscReal *area, PetscReal *vol) {
    PetscReal *c = NULL;
    ablate::levelSet::Utilities::VertexLevelSet_LS(dm, p, c0, nIn, &c);

    ablate::levelSet::Utilities::VOF(dm, p, c, vof, area, vol);  // Do the actual calculation.

    PetscFree(c) >> ablate::utilities::PetscUtilities::checkError;
}

// Returns the VOF for a given cell using an analytic level set equation
// Refer to "Quadrature rules for triangular and tetrahedral elements with generalized functions"
void ablate::levelSet::Utilities::VOF(DM dm, PetscInt p, const std::shared_ptr<ablate::mathFunctions::MathFunction> &phi, PetscReal *vof, PetscReal *area, PetscReal *vol) {
    PetscInt dim, Nc, nVerts, i;
    PetscReal *c = NULL, *coords = NULL;
    const PetscScalar *array;
    PetscBool isDG;

    DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

    // Coordinates of the cell vertices
    DMPlexGetCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;

    // Number of vertices
    nVerts = Nc / dim;

    PetscMalloc1(nVerts, &c) >> ablate::utilities::PetscUtilities::checkError;

    // The level set value of each vertex. This assumes that the interface is a line/plane
    //    with the given unit normal.
    for (i = 0; i < nVerts; ++i) {
        c[i] = phi->Eval(&coords[i * dim], dim, 0.0);
    }

    DMPlexRestoreCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;

    ablate::levelSet::Utilities::VOF(dm, p, c, vof, area, vol);  // Do the actual calculation.

    PetscFree(c) >> ablate::utilities::PetscUtilities::checkError;
}

// Return the VOF in a cell where the level set is defined at vertices
void ablate::levelSet::Utilities::VOF(std::shared_ptr<ablate::domain::SubDomain> subDomain, PetscInt cell, const ablate::domain::Field *lsField, PetscReal *vof, PetscReal *area, PetscReal *vol) {
    DM dm = subDomain->GetFieldDM(*lsField);
    Vec vec = subDomain->GetVec(*lsField);
    const PetscScalar *array;
    PetscReal *c;

    PetscInt nv, *verts;
    DMPlexCellGetVertices(dm, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
    DMGetWorkArray(dm, nv, MPI_REAL, &c);

    VecGetArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
    for (PetscInt i = 0; i < nv; ++i) {
        const PetscReal *val = nullptr;
        xDMPlexPointLocalRead(dm, verts[i], lsField->id, array, &val) >> ablate::utilities::PetscUtilities::checkError;
        c[i] = *val;
    }
    VecRestoreArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;

    ablate::levelSet::Utilities::VOF(dm, cell, c, vof, area, vol);

    DMRestoreWorkArray(dm, nv, MPI_REAL, &c);
    DMPlexCellRestoreVertices(dm, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
}



//static void SaveVertexData(DM dm, Vec vec, const char fname[255], const ablate::domain::Field *field, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

//  ablate::domain::Range range;
//  const PetscReal    *array, *val;
//  PetscInt      dim = subDomain->GetDimensions();
//  MPI_Comm      comm = PetscObjectComm((PetscObject)dm);
//  int rank, size;
//  MPI_Comm_size(comm, &size) >> ablate::utilities::MpiUtilities::checkError;
//  MPI_Comm_rank(comm, &rank) >> ablate::utilities::MpiUtilities::checkError;

//  ablate::domain::GetRange(dm, nullptr, 0, range);

//  VecGetArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;

//  for (PetscInt r = 0; r < size; ++r) {
//    if ( rank==r ) {

//      FILE *f1;
//      if ( rank==0 ) f1 = fopen(fname, "w");
//      else f1 = fopen(fname, "a");

//      for (PetscInt v = range.start; v < range.end; ++v) {
//        PetscInt vert = range.points ? range.points[v] : v;
//        PetscScalar *coords;

//        DMPlexPointLocalFieldRead(dm, vert, field->id, array, &val) >> ablate::utilities::PetscUtilities::checkError;

//        DMPlexVertexGetCoordinates(dm, 1, &vert, &coords);

//        for (PetscInt d = 0; d < dim; ++d) {
//          fprintf(f1, "%+.16e\t", coords[d]);
//        }
//        fprintf(f1, "%+.16e\n", *val);

//        DMPlexVertexRestoreCoordinates(dm, 1, &vert, &coords);
//      }

//      fclose(f1);
//    }
//    MPI_Barrier(comm);
//  }


//  VecRestoreArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
//  ablate::domain::RestoreRange(range);
//}

//static void SaveVertexData(const char fname[255], const ablate::domain::Field *field, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

//  Vec           vec = subDomain->GetVec(*field);
//  DM            dm  = subDomain->GetFieldDM(*field);
//  SaveVertexData(dm, vec, fname, field, subDomain);
//}


void SaveCellData(const char fname[255], const ablate::domain::Field *field, PetscInt Nc, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

  ablate::domain::Range range;
  const PetscScalar    *array = nullptr;
  Vec           vec = subDomain->GetVec(*field);
  DM            dm  = subDomain->GetFieldDM(*field);
  PetscInt      dim = subDomain->GetDimensions();
  MPI_Comm      comm = PetscObjectComm((PetscObject)dm);
  int rank, size;
  MPI_Comm_size(comm, &size) >> ablate::utilities::MpiUtilities::checkError;
  MPI_Comm_rank(comm, &rank) >> ablate::utilities::MpiUtilities::checkError;

  subDomain->GetCellRange(nullptr, range);

  VecGetArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;



  for (PetscInt r = 0; r < size; ++r) {
    if ( rank==r ) {

      FILE *f1;
      if ( rank==0 ) f1 = fopen(fname, "w");
      else f1 = fopen(fname, "a");

      for (PetscInt c = range.start; c < range.end; ++c) {
        PetscInt cell = range.points ? range.points[c] : c;

        if (ablate::levelSet::Utilities::ValidCell(dm, c)) {

          PetscReal x0[3];
          DMPlexComputeCellGeometryFVM(dm, cell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

          const PetscScalar *val = nullptr;
          DMPlexPointLocalFieldRead(dm, cell, field->id, array, &val) >> ablate::utilities::PetscUtilities::checkError;

          for (PetscInt d = 0; d < dim; ++d) {
            fprintf(f1, "%+f\t", x0[d]);
          }

          for (PetscInt i = 0; i < Nc; ++i) {
            fprintf(f1, "%+f\t", val[i]);
          }
          fprintf(f1, "\n");
        }
      }
      fclose(f1);
    }

    MPI_Barrier(PETSC_COMM_WORLD);
  }


  VecRestoreArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
  ablate::domain::RestoreRange(range);
}

// vofField - Field containing the cell volume-of-fluid
// cellNormalField - Unit normals at cell-centers. This is pre-computed and an input
// accumField - The accumulator field for number of elements
// lsField - The updated level-set values at vertices
static void CutCellLevelSetValues(std::shared_ptr<ablate::domain::SubDomain> subDomain, ablate::domain::Range cellRange, ablate::domain::Range vertRange, ablate::domain::ReverseRange reverseVertRange, PetscInt *cellMask, const ablate::domain::Field *vofField, const ablate::domain::Field *cellNormalField, const ablate::domain::Field *accumField, const ablate::domain::Field *lsField) {

  DM              solDM = subDomain->GetDM();
  DM              auxDM = subDomain->GetAuxDM();
  Vec             solVec = subDomain->GetSolutionVector();
  Vec             auxVec = subDomain->GetAuxVector();
  const PetscInt  vofID = vofField->id;
  const PetscInt  normalID = cellNormalField->id;
  const PetscInt  lsID = lsField->id;

  Vec workVec = nullptr;
  PetscScalar *workArray = nullptr;
  DMGetLocalVector(auxDM, &workVec);



  const PetscScalar *solArray = nullptr;
  PetscScalar *auxArray = nullptr;

  VecZeroEntries(workVec);

  VecGetArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(workVec, &workArray) >> ablate::utilities::PetscUtilities::checkError;

int rank;
MPI_Comm_rank(PETSC_COMM_WORLD, &rank) >> ablate::utilities::MpiUtilities::checkError;
char fname[255];
sprintf(fname, "proc%d.txt",rank);
FILE *f1 = fopen(fname, "w");
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

    // Only worry about cut-cells
    if ( cellMask[c] == 1 ) {

      PetscInt cell = cellRange.GetPoint(c);
PetscReal x0[3];
DMPlexComputeCellGeometryFVM(auxDM, cell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;
fprintf(f1, "%f\t%f\n", x0[0], x0[1]);
      // The VOF for the cell
      const PetscScalar *vofVal = nullptr;
      xDMPlexPointLocalRead(solDM, cell, vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

      // The pre-computed cell-centered normal
      const PetscScalar *n = nullptr;
      xDMPlexPointLocalRead(auxDM, cell, normalID, auxArray, &n) >> ablate::utilities::PetscUtilities::checkError;

      PetscInt nv, *verts;
      DMPlexCellGetVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

      PetscReal *lsVertVals = NULL;
      DMGetWorkArray(auxDM, nv, MPIU_REAL, &lsVertVals) >> ablate::utilities::PetscUtilities::checkError;

      // Level set values at the vertices
      ablate::levelSet::Utilities::VertexLevelSet_VOF(auxDM, cell, *vofVal, n, &lsVertVals);

      for (PetscInt v = 0; v < nv; ++v) {
        PetscScalar *lsVal = nullptr;
        xDMPlexPointLocalRef(auxDM, verts[v], lsID, workArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;
        *lsVal += lsVertVals[v];

        PetscScalar *lsCount = nullptr;
        xDMPlexPointLocalRef(auxDM, verts[v], accumField->id, workArray, &lsCount) >> ablate::utilities::PetscUtilities::checkError;
        *lsCount += 1.0;
      }

      DMRestoreWorkArray(auxDM, nv, MPIU_REAL, &lsVertVals) >> ablate::utilities::PetscUtilities::checkError;
      DMPlexCellRestoreVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

    }
  }

  // This is no longer needed
  VecRestoreArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;


  Vec workVecGlobal;
  DMGetGlobalVector(auxDM, &workVecGlobal) >> ablate::utilities::PetscUtilities::checkError;
  DMLocalToGlobal(auxDM, workVec, ADD_VALUES, workVecGlobal) >> ablate::utilities::PetscUtilities::checkError;
  DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreGlobalVector(auxDM, &workVecGlobal) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

    PetscInt vert = vertRange.GetPoint(v);
    const PetscScalar *lsCount = nullptr;
    xDMPlexPointLocalRead(auxDM, vert, accumField->id, workArray, &lsCount) >> ablate::utilities::PetscUtilities::checkError;


    if ( *lsCount > 0 ) {

      PetscReal *lsVal = nullptr;
      xDMPlexPointLocalRef(auxDM, vert, lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;

      const PetscScalar *lsSum = nullptr;
      xDMPlexPointLocalRead(auxDM, vert, lsID, workArray, &lsSum) >> ablate::utilities::PetscUtilities::checkError;

//if(fabs(x0[0]-0.1)<0.001 && fabs(x0[1]-1)<0.001) printf("%+f\t%+f\n", *lsSum, *lsCount);

      *lsVal = (*lsSum)/(*lsCount);
    }
  }
fclose(f1);
exit(0);
  VecRestoreArray(workVec, &workArray) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreLocalVector(auxDM, &workVec) >> ablate::utilities::PetscUtilities::checkError;

  VecRestoreArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;

  subDomain->UpdateAuxLocalVector();


}




//struct reinitCTX {
//    std::shared_ptr<ablate::domain::rbf::RBF> rbf;
//};
//static Parameters parameters{};
#include "domain/RBF/ga.hpp"
#include "domain/RBF/hybrid.hpp"
#include "domain/RBF/imq.hpp"
#include "domain/RBF/mq.hpp"
#include "domain/RBF/phs.hpp"
#include "domain/RBF/rbf.hpp"
static std::shared_ptr<ablate::domain::rbf::RBF> rbf = nullptr;

// Temporary for the review
static PetscInt **cellNeighbors = nullptr, *numberNeighbors = nullptr;

// Make sure that the work is being done on valid cells and not ghost cells
bool ablate::levelSet::Utilities::ValidCell(DM dm, PetscInt p) {
    DMPolytopeType ct;
    DMPlexGetCellType(dm, p, &ct) >> ablate::utilities::PetscUtilities::checkError;

    return (ct < 12);
}



//vofField: cell-based field containing the target volume-of-fluid
//lsField: vertex-based field for level set values
//normalField: cell-based vector field containing normals
//curvField: cell-based vector field containing curvature
void ablate::levelSet::Utilities::Reinitialize(std::shared_ptr<ablate::domain::SubDomain> subDomain, Vec solVec, const ablate::domain::Field *vofField, const PetscInt nLevels, const ablate::domain::Field *lsField, const ablate::domain::Field *vertexNormalField, const ablate::domain::Field *cellNormalField, const ablate::domain::Field *curvField) {

  // Note: Need to write a unit test where the vof and ls fields aren't in the same DM, e.g. one is a SOL vector and one is an AUX vector.

  // Make sure that all of the fields are in the correct locations.
  if ( vofField->location != ablate::domain::FieldLocation::SOL ){
    throw std::runtime_error("VOF field must be in SOL");
  }

  if ( lsField->location != ablate::domain::FieldLocation::AUX ){
    throw std::runtime_error("Level set field must be in AUX");
  }

  if ( vertexNormalField->location != ablate::domain::FieldLocation::AUX ){
    throw std::runtime_error("Vertex Normal field must be in AUX");
  }

  if ( cellNormalField->location != ablate::domain::FieldLocation::AUX ){
    throw std::runtime_error("Cell Normal field must be in AUX");
  }

  if ( curvField->location != ablate::domain::FieldLocation::AUX ){
    throw std::runtime_error("Curvature Field field must be in AUX");
  }

SaveCellData("vof.txt", vofField, 1, subDomain);
  PetscReal         h = 0.0;
  const PetscInt    dim = subDomain->GetDimensions();   // VOF and LS subdomains must have the same dimension. Can't think of a reason they wouldn't.
  PetscInt          *vertMask = nullptr, *cellMask = nullptr;
  DM                solDM = subDomain->GetDM();
  DM                auxDM = subDomain->GetAuxDM();
//  Vec               solVec = subDomain->GetSolutionVector();
  Vec               auxVec = subDomain->GetAuxVector();
  const PetscScalar *solArray = nullptr;
  PetscScalar       *auxArray = nullptr;
  const PetscInt    lsID = lsField->id, vofID = vofField->id, cellNormalID = cellNormalField->id;

  DMPlexGetMinRadius(solDM, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size

  /***********************************************************************************************/
  // THIS IS TEMPORARY AND NEEDS TO BE MOVED TO THE YAML FILE OR SOMETHING ELSE AFTER THE REVIEW
  /***********************************************************************************************/
  if ( rbf==nullptr ) {
    PetscInt polyAug = 3;
    bool doesNotHaveDerivatives = false;
    bool doesNotHaveInterpolation = false;
    bool returnNeighborVertices = true;
    rbf = std::make_shared<ablate::domain::rbf::GA>(polyAug, 0.1*h, doesNotHaveDerivatives, doesNotHaveInterpolation, returnNeighborVertices);

    rbf->Setup(subDomain);       // This causes issues (I think)
    rbf->Initialize();  //         Initialize
  }

  VecGetArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;


  ablate::domain::Range cellRange, vertRange;
  subDomain->GetCellRange(nullptr, cellRange);
  subDomain->GetRange(nullptr, 0, vertRange);

  if (cellNeighbors==nullptr) {
    PetscMalloc1(cellRange.end - cellRange.start, &cellNeighbors) >> ablate::utilities::PetscUtilities::checkError;
    cellNeighbors -= cellRange.start;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      cellNeighbors[c] = nullptr;
    }
  }

  if (numberNeighbors==nullptr) {
    PetscMalloc1(cellRange.end - cellRange.start, &numberNeighbors) >> ablate::utilities::PetscUtilities::checkError;
    numberNeighbors -= cellRange.start;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      numberNeighbors[c] = 0;
    }
  }


  // Get the point->index mapping for cells
  ablate::domain::ReverseRange reverseVertRange = ablate::domain::ReverseRange(vertRange);
  ablate::domain::ReverseRange reverseCellRange = ablate::domain::ReverseRange(cellRange);

  // Pull some work arrays to store the mask information
  DMGetWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &vertMask) >> ablate::utilities::PetscUtilities::checkError;
  PetscArrayzero(vertMask, vertRange.end - vertRange.start);
  vertMask -= vertRange.start; // offset so that we can use start->end

  DMGetWorkArray(solDM, cellRange.end - cellRange.start, MPIU_INT, &cellMask) >> ablate::utilities::PetscUtilities::checkError;
  PetscArrayzero(cellMask, cellRange.end - cellRange.start);
  cellMask -= cellRange.start; // offset so that we can use start->end


  PetscInt *closestCell = nullptr;
  DMGetWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &closestCell) >> ablate::utilities::PetscUtilities::checkError;
  PetscArrayzero(closestCell, vertRange.end - vertRange.start);
  closestCell -= vertRange.start; // offset so that we can use start->end

  // Setup the cut-cell locations and the initial unit normal estimate

int rank;
MPI_Comm_rank(PETSC_COMM_WORLD, &rank) >> ablate::utilities::MpiUtilities::checkError;
char fname[255];
sprintf(fname, "proc%d.txt",rank);
FILE *f1 = fopen(fname, "w");

  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

    PetscInt cell = cellRange.GetPoint(c);

    if (ablate::levelSet::Utilities::ValidCell(solDM, cell)) {

      const PetscScalar *vofVal = nullptr;
      xDMPlexPointLocalRead(solDM, cell, vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

      // Set the initial normal vector equal to zero. This will ensure that any cells downwind during the PDE update won't have a contribution
      PetscScalar *n = nullptr;
      xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &n);
      for (PetscInt d = 0; d < dim; ++d ) n[d] = 0.0;

      // Only worry about cut-cells
      if ( ((*vofVal) > ablate::utilities::Constants::small) && ((*vofVal) < (1.0 - ablate::utilities::Constants::small)) ) {


PetscReal x[3];
DMPlexComputeCellGeometryFVM(solDM, cell, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
fprintf(f1, "%f\t%f\n", x[0], x[1]);

        cellMask[c] = 1;  // Mark as a cut-cell

        // Compute an estimate of the unit normal at the cell-center
        DMPlexCellGradFromCell(solDM, cell, solVec, vofID, 0, n);
        ablate::utilities::MathUtilities::NormVector(dim, n);
        for (PetscInt d = 0; d < dim; ++d) n[d] *= -1.0;

        // Mark all vertices of this cell as associated with a cut-cell
        PetscInt nv, *verts;
        DMPlexCellGetVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
        for (PetscInt v = 0; v < nv; ++v) {
          PetscInt vert_i = reverseVertRange.GetIndex(verts[v]);
          vertMask[vert_i] = 1;
        }
        DMPlexCellRestoreVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
      }
    }
  }
fclose(f1);
exit(0);

  // Temporary level-set work array to store old or new values, as appropriate
  PetscScalar *tempLS;
  DMGetWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_SCALAR, &tempLS) >> ablate::utilities::PetscUtilities::checkError;
  tempLS -= vertRange.start;

  PetscReal maxDiff = 1.0;
  PetscInt iter = 0;

  MPI_Comm auxCOMM = PetscObjectComm((PetscObject)auxDM);


  while ( maxDiff > 1e-3*h && iter<20 ) {

    ++iter;

    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
      if (vertMask[v]==1) {
        PetscInt vert = vertRange.GetPoint(v);
        const PetscReal *oldLS = nullptr;
        xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &oldLS) >> ablate::utilities::PetscUtilities::checkError;
        tempLS[v] = *oldLS;
      }
    }

    // This updates the lsField by taking the average vertex values necessary to match the VOF in cutcells
    CutCellLevelSetValues(subDomain, cellRange, vertRange, reverseVertRange, cellMask, vofField, cellNormalField, vertexNormalField, lsField);

    // Update the normals
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      if (cellMask[c] == 1) {
        PetscInt cell = cellRange.GetPoint(c);
        PetscScalar *n = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &n);
        DMPlexCellGradFromVertex(auxDM, cell, auxVec, lsID, 0, n) >> ablate::utilities::PetscUtilities::checkError;
        ablate::utilities::MathUtilities::NormVector(dim, n);
      }
    }


    // Now compute the difference on this processor
    maxDiff = -1.0;
    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

      if (vertMask[v] == 1) {
        PetscInt vert = vertRange.GetPoint(v);
        const PetscReal *newLS = nullptr;
        xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &newLS) >> ablate::utilities::PetscUtilities::checkError;

        maxDiff = PetscMax(maxDiff, PetscAbsReal(tempLS[v] - *newLS));

      }
    }
    // Get the maximum change across all processors. This also acts as a sync point
    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, auxCOMM);

    PetscPrintf(PETSC_COMM_WORLD, "Cut Cells %d: %+e\n", iter, maxDiff) >> ablate::utilities::PetscUtilities::checkError;
  }
exit(0);
//  SaveCellData("normal.txt", cellNormalField, dim, subDomain);
//  SaveVertexData("ls0.txt", lsField, subDomain);

  // Set the vertices far away as the largest possible value in the domain with the appropriate sign.
  // This is done after the determination of cut-cells so that all vertices associated with cut-cells have been marked.
  PetscReal gMin[3], gMax[3], maxDist = -1.0;
  DMGetBoundingBox(auxDM, gMin, gMax) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt d = 0; d < dim; ++d) {
    maxDist = PetscMax(maxDist, gMax[d] - gMin[d]);
  }
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);

    // Only worry about cells to far away
    if ( cellMask[c] == 0 && ablate::levelSet::Utilities::ValidCell(solDM, cell)) {
      const PetscScalar *vofVal = nullptr;
      xDMPlexPointLocalRead(solDM, cell, vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

      PetscReal sgn = PetscSignReal(0.5 - (*vofVal));

      PetscInt nv, *verts;
      DMPlexCellGetVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

      for (PetscInt v = 0; v < nv; ++v) {
        PetscInt id = reverseVertRange.GetIndex(verts[v]);
        if (vertMask[id] == 0) {
          PetscScalar *lsVal = nullptr;
          xDMPlexPointLocalRef(auxDM, verts[v], lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;
          *lsVal = sgn*maxDist;
        }
      }
      DMPlexCellRestoreVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
    }
  }

  // Now mark all of the necessary neighboring vertices. Note that this can't be put into the previous loop as all of the vertices
  //    for the cut-cells won't be known yet.
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    // Only worry about cut-cells
    if ( cellMask[c] == 1 ) {

      PetscInt cutCell = cellRange.GetPoint(c);

      // Center of the cell
      PetscReal x0[3];
      DMPlexComputeCellGeometryFVM(solDM, cutCell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

      // Get the level-set value at the cell-center
      PetscInt nv, *verts;
      DMPlexCellGetVertices(solDM, cutCell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
      PetscScalar c0 = 0.0;
      for (PetscInt v = 0; v < nv; ++v) {
        PetscScalar *lsVal = nullptr;
        xDMPlexPointLocalRef(auxDM, verts[v], lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;
        c0 += *lsVal;
      }
      c0 /= nv;
      DMPlexCellRestoreVertices(solDM, cutCell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

      if (cellNeighbors[c]==nullptr) {

        PetscInt nCellsNew, *newCells;

        DMPlexGetNeighbors(solDM, cutCell, nLevels, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCellsNew, &newCells) >> ablate::utilities::PetscUtilities::checkError;
        PetscMalloc1(nCellsNew, &cellNeighbors[c]);
        PetscArraycpy(cellNeighbors[c], newCells, nCellsNew);
        numberNeighbors[c] = nCellsNew;
        DMPlexRestoreNeighbors(solDM, cutCell, nLevels, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCellsNew, &newCells) >> ablate::utilities::PetscUtilities::checkError;

      }
      const PetscInt nCells = numberNeighbors[c];
      const PetscInt *cells = cellNeighbors[c];

      for (PetscInt i = 0; i < nCells; ++i) {

        PetscInt cellID = reverseCellRange.GetIndex(cells[i]);
        if (cellMask[cellID] == 0 && ablate::levelSet::Utilities::ValidCell(solDM, cells[i])) {
          cellMask[cellID] = 2; // Mark as a cell where cell-centered gradients are needed
        }

        if (cellMask[cellID]==2) {

          PetscInt nv, *verts;
          DMPlexCellGetVertices(solDM, cells[i], &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

          PetscScalar *coords;
          DMPlexVertexGetCoordinates(solDM, nv, verts, &coords) >> ablate::utilities::PetscUtilities::checkError;

          for (PetscInt v = 0; v < nv; ++v) {
            PetscInt id = reverseVertRange.GetIndex(verts[v]);
            if (vertMask[id] == 0) {
              vertMask[id] = 2;  // Mark the vertex as associated with a cell that needs a cell-center gradient that is NOT associated with a cut-cell.
            }

            if (vertMask[id]==2) { // This is done separately from the prior if-statement so that we can compute the minimum distance

              // As an initial guess at the signed-distance function use the distance from the cut-cell center to this vertex
              PetscReal dist = 0.0;
              for (PetscInt d = 0; d < dim; ++d) {
                dist += PetscSqr(x0[d] - coords[v*dim + d]);
              }
              dist = c0 + PetscSqrtReal(dist);

              PetscScalar *lsVal = nullptr;
              xDMPlexPointLocalRef(auxDM, verts[v], lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;

              if (dist < PetscAbs(*lsVal)) {
                PetscScalar *vofVal = nullptr;
                xDMPlexPointLocalRead(solDM, cells[i], vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;
                PetscReal sgn = (*vofVal < 0.5 ? +1.0 : -1.0);
                *lsVal = sgn*dist;
                closestCell[id] = cutCell;
              }
            }
          }
          DMPlexCellRestoreVertices(solDM, cells[i], &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
          DMPlexVertexRestoreCoordinates(solDM, nv, verts, &coords) >> ablate::utilities::PetscUtilities::checkError;
        }
      }
    }
  }
exit(0);
  VecRestoreArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;

  subDomain->UpdateAuxLocalVector();
//  SaveVertexData("ls1.txt", lsField, subDomain);

  const PetscInt vertexNormalID = vertexNormalField->id;
  const PetscInt curvID = curvField->id;

  maxDiff = 1.0;
  iter = 0;
  while (maxDiff>1e-2 && iter<10) {
    ++iter;


    // Determine the current gradient at cells that need updating
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      if (cellMask[c] > 1) {
        PetscInt cell = cellRange.GetPoint(c);
        PetscReal *g = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &g) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexCellGradFromVertex(auxDM, cell, auxVec, lsID, 0, g);
      }
    }

    maxDiff = -1.0;

    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
      if (vertMask[v] > 1) {
        PetscInt vert = vertRange.GetPoint(v);
        PetscReal g[dim];
        PetscReal *phi = nullptr;

        xDMPlexPointLocalRef(auxDM, vert, lsID, auxArray, &phi) >> ablate::utilities::PetscUtilities::checkError;

        DMPlexVertexGradFromVertex(auxDM, vert, auxVec, lsID, 0, g) >> ablate::utilities::PetscUtilities::checkError;

        PetscReal s = PetscSignReal(*phi);

        VertexUpwindGrad(auxDM, auxArray, cellNormalID, vert, s, g);

        PetscReal nrm = ablate::utilities::MathUtilities::MagVector(dim, g);

        // If the vertex is near the edge of a domain when running in parallel the gradient will be zero, so ignore it.
        if ( nrm > 0.0 ) {

          PetscReal mag = nrm - 1.0;

          *phi -= h*s*mag;

          maxDiff = PetscMax(maxDiff, PetscAbsReal(mag));
        }
      }
    }

     // Get the maximum change across all processors. This also acts as a sync point
    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, auxCOMM);

    subDomain->UpdateAuxLocalVector();

//    PetscPrintf(PETSC_COMM_WORLD, "%3d: %e\n", iter, maxDiff);
  }



//SaveVertexData("ls2.txt", lsField, subDomain);

//  // Try a smoother
//  // Mark the outer layer as fixed in the smoothing step
//  for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
//    if (vertMask[v]==2) {
//      PetscInt vert = vertRange.GetPoint(v);
//      PetscInt nVerts, *verts;

//      DMPlexGetNeighbors(auxDM, vert, 1, -1.0, -1.0, PETSC_TRUE, PETSC_TRUE, &nVerts, &verts) >> ablate::utilities::PetscUtilities::checkError;
//      for (PetscInt i = 0; i < nVerts; ++i) {
//        PetscInt id = reverseVertRange.GetIndex(verts[i]);
//        if (vertMask[id] == 0) {
//          vertMask[v] = 3;
//        }
//      }
//      DMPlexRestoreNeighbors(auxDM, vert, 1, -1.0, -1.0, PETSC_TRUE, PETSC_TRUE, &nVerts, &verts) >> ablate::utilities::PetscUtilities::checkError;
//    }
//  }

//  for(iter=0;iter<10;++iter){

//    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
//      if (vertMask[v]==1 || vertMask[v]==2 ) {
//        PetscInt vert = vertRange.GetPoint(v);
//        PetscInt nVerts, *verts;


//        DMPlexGetNeighbors(auxDM, vert, 1, -1.0, -1.0, PETSC_TRUE, PETSC_TRUE, &nVerts, &verts) >> ablate::utilities::PetscUtilities::checkError;

//        tempLS[v] = 0.0;
//        PetscInt nv = 0;
//        for (PetscInt i = 0; i < nVerts; ++i) {
//          PetscInt id = reverseVertRange.GetIndex(verts[i]);
////          if (verts[i] != vert && vertMask[id] > 0) {
//          if (vertMask[id] > 0) {
//            ++nv;
//            const PetscReal *phi = nullptr;
//            xDMPlexPointLocalRead(auxDM, verts[i], lsID, auxArray, &phi);
//            tempLS[v] += *phi;
//          }
//        }


//        tempLS[v] /= nv;
////        const PetscReal *phi = nullptr;
////        xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &phi);
////        if (nv>0) {
////          tempLS[v] = 0.5*phi[0] + 0.5*tempLS[v]/(nv);
////        }
////        else {
////          tempLS[v] = phi[0];
////        }


//        DMPlexRestoreNeighbors(auxDM, vert, 1, -1.0, -1.0, PETSC_FALSE, PETSC_TRUE, &nVerts, &verts) >> ablate::utilities::PetscUtilities::checkError;

//      }
//    }

//    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
//      if (vertMask[v]==1 || vertMask[v]==2) {
//        PetscInt vert = vertRange.GetPoint(v);
//        PetscReal *phi = nullptr;
//        xDMPlexPointLocalRef(auxDM, vert, lsID, auxArray, &phi) >> ablate::utilities::PetscUtilities::checkError;
//        *phi = tempLS[v];
//      }
//    }
//  }

  // Calculate unit normal vector based on the updated level set values at the vertices
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

    if (cellMask[c] > 0) {
      PetscInt cell = cellRange.GetPoint(c);
      PetscScalar *n = nullptr;
      xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &n);
      DMPlexCellGradFromVertex(auxDM, cell, auxVec, lsID, 0, n) >> ablate::utilities::PetscUtilities::checkError;
      ablate::utilities::MathUtilities::NormVector(dim, n);
    }
  }

  // Calculate vertex-based unit normal vector based on the updated level set values at the vertices
  for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

    if (vertMask[v] > 0) {
      PetscInt vert = vertRange.GetPoint(v);

      PetscScalar *n = nullptr;
      xDMPlexPointLocalRef(auxDM, vert, vertexNormalID, auxArray, &n);
      DMPlexVertexGradFromVertex(auxDM, vert, auxVec, lsID, 0, n) >> ablate::utilities::PetscUtilities::checkError;
      ablate::utilities::MathUtilities::NormVector(dim, n);
    }
  }

  // Curvature
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

    if (cellMask[c] > 0) {

      PetscInt cell = cellRange.GetPoint(c);

      PetscScalar *H = nullptr;
      xDMPlexPointLocalRef(auxDM, cell, curvID, auxArray, &H);

      *H = ablate::levelSet::geometry::Curvature(rbf, lsField, cell);

//      *H = 0;
//      for (PetscInt d = 0; d < dim; ++d) {
//        PetscReal g[dim];
////        DMPlexCellGradFromCell(auxDM, cell, auxVec, cellNormalID, d, g) >> ablate::utilities::PetscUtilities::checkError;
//        DMPlexCellGradFromVertex(auxDM, cell, auxVec, vertexNormalID, d, g) >> ablate::utilities::PetscUtilities::checkError;
//        *H += g[d];
//      }
    }
  }

  subDomain->UpdateAuxLocalVector();

//  SaveCellData("curv0.txt", curvField, 1, subDomain);

  // Extension
  PetscInt vertexCurvID = lsID; // Store the vertex curvatures in the work vec at the same location as the level-set
  Vec workVec, workVecGlobal;
  PetscScalar *workArray = nullptr;
  DMGetLocalVector(auxDM, &workVec);
  DMGetGlobalVector(auxDM, &workVecGlobal);

  VecGetArray(workVec, &workArray);
  for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
    if (vertMask[v] > 0) {
      PetscInt vert = vertRange.GetPoint(v);

      PetscReal *vertexH = nullptr;
      xDMPlexPointLocalRef(auxDM, vert, vertexCurvID, workArray, &vertexH);

      if (vertMask[v]==1) {

        PetscInt nc = 0;
        *vertexH = 0.0;

        PetscInt nCells, *cells;
        DMPlexVertexGetCells(auxDM, vert, &nCells, &cells);
        for (PetscInt c = 0; c < nCells; ++c) {

          PetscInt id = reverseVertRange.GetIndex(cells[c]);

          if (cellMask[id] == 1) {
            ++nc;
            const PetscReal *cellH = nullptr;
            xDMPlexPointLocalRead(auxDM, cells[c], curvID, auxArray, &cellH);

            *vertexH += *cellH;
          }
        }

        *vertexH /= nc;

        DMPlexVertexRestoreCells(auxDM, v, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
      }
      else {
        PetscReal *cellH = nullptr;
        xDMPlexPointLocalRef(auxDM, closestCell[v], curvID, auxArray, &cellH);
        *vertexH = *cellH;
      }
    }
  }
  VecRestoreArray(workVec, &workArray);

//  SaveVertexData(auxDM, workVec, "vertexH0.txt", lsField, subDomain);

  maxDiff = 1.0;
  iter = 0;
  while ( maxDiff>1e-2 && iter<10) {
    ++iter;

    VecGetArray(workVec, &workArray);

    // Curvature gradient at the cell-center
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      if (cellMask[c] > 0) {
        PetscInt cell = cellRange.GetPoint(c);
        PetscScalar *g = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, cellNormalID, workArray, &g) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexCellGradFromVertex(auxDM, cell, workVec, vertexCurvID, 0, g);
      }
    }

    maxDiff = -1.0;


    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
      if (vertMask[v] > 0) {
        PetscInt vert = vertRange.GetPoint(v);
        PetscReal g[dim];
        const PetscReal *phi = nullptr, *n = nullptr;
        xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &phi) >> ablate::utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRead(auxDM, vert, vertexNormalID, auxArray, &n) >> ablate::utilities::PetscUtilities::checkError;
        for (PetscInt d = 0; d < dim; ++d) g[d] = n[d];

        VertexUpwindGrad(auxDM, workArray, cellNormalID, vert, PetscSignReal(*phi), g);

        PetscReal dH = 0.0;
        for (PetscInt d = 0; d < dim; ++d) dH += g[d]*n[d];


        PetscReal *H = nullptr;
        xDMPlexPointLocalRef(auxDM, vert, vertexCurvID, workArray, &H);

        PetscReal s = *phi/PetscSqrtReal(PetscSqr(*phi) + h*h);

        *H -= h*s*dH;

        maxDiff = PetscMax(maxDiff, PetscAbsReal(dH/(*H)));
      }
    }
    VecRestoreArray(workVec, &workArray);

    DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;

     // Get the maximum change across all processors. This also acts as a sync point
    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, auxCOMM);

//    PetscPrintf(PETSC_COMM_WORLD, "Extension %3d: %e\n", iter, maxDiff);

  }

  // Now set the curvature at the cell-center via averaging
  VecGetArray(workVec, &workArray);
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    if (cellMask[c] > 0) {
      PetscInt cell = cellRange.GetPoint(c);

      PetscScalar *cellH = nullptr;
      xDMPlexPointLocalRef(auxDM, cell, curvID, auxArray, &cellH) >> utilities::PetscUtilities::checkError;

      *cellH = 0.0;

      PetscInt nv, *verts;
      DMPlexCellGetVertices(auxDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
      for (PetscInt i = 0; i < nv; ++i) {
        const PetscReal *H;
        xDMPlexPointLocalRead(auxDM, verts[i], vertexCurvID, workArray, &H) >> utilities::PetscUtilities::checkError;
        *cellH += *H;
      }
      *cellH /= nv;

//*cellH = 1.0;

      DMPlexCellRestoreVertices(auxDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
    }
  }
  VecRestoreArray(workVec, &workArray);

  subDomain->UpdateAuxLocalVector();


//  SaveCellData("curv.txt", curvField, 1, subDomain);
//exit(0);
//  SaveVertexData(auxDM, workVec, "vertexH1.txt", lsField, subDomain);
  DMRestoreLocalVector(auxDM, &workVec) >> utilities::PetscUtilities::checkError;
  DMRestoreGlobalVector(auxDM, &workVecGlobal) >> utilities::PetscUtilities::checkError;


  closestCell += vertRange.start;
  DMGetWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &closestCell) >> ablate::utilities::PetscUtilities::checkError;

  // Cleanup all memory
  tempLS += vertRange.start;
  DMRestoreWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_SCALAR, &tempLS) >> ablate::utilities::PetscUtilities::checkError;
  vertMask += vertRange.start; // Reset the offset, otherwise DMRestoreWorkArray will return unexpected results
  DMRestoreWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &vertMask) >> ablate::utilities::PetscUtilities::checkError;
  cellMask += cellRange.start; // Reset the offset, otherwise DMRestoreWorkArray will return unexpected results
  DMRestoreWorkArray(solDM, cellRange.end - cellRange.start, MPIU_INT, &cellMask) >> ablate::utilities::PetscUtilities::checkError;

  subDomain->RestoreRange(vertRange);

  VecRestoreArray(auxVec, &auxArray) >> utilities::PetscUtilities::checkError;

}
