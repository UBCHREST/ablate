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


void DMPlexVertexDivFromCellUpwind(DM dm, const PetscInt v, Vec data, const PetscInt fID, const PetscReal s, const PetscReal g[], PetscReal *div) {

    const PetscScalar *dataArray;
    PetscInt cStart, cEnd;
    PetscInt dim;
    PetscInt nStar, *star = NULL;

    DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

    PetscReal x0[dim];
    DMPlexComputeCellGeometryFVM(dm, v, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

    DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> ablate::utilities::PetscUtilities::checkError;
    VecGetArrayRead(data, &dataArray) >> ablate::utilities::PetscUtilities::checkError;

    *div = 0.0;
    PetscReal totalVol = 0.0;

    // Everything using this vertex
    DMPlexGetTransitiveClosure(dm, v, PETSC_FALSE, &nStar, &star) >> ablate::utilities::PetscUtilities::checkError;
    for (PetscInt st = 0; st < nStar * 2; st += 2) {
        if (star[st] >= cStart && star[st] < cEnd) {  // It's a cell

            PetscReal x[dim];
            DMPlexComputeCellGeometryFVM(dm, star[st], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;

            PetscReal dot = 0.0;
            for (PetscInt d = 0; d < dim; ++d) {
              dot += g[d]*(x0[d] - x[d]);
            }

            if (s*dot>=0.0) {
              // Surface area normal
              PetscScalar N[3];
              DMPlexCornerSurfaceAreaNormal(dm, v, star[st], N) >> ablate::utilities::PetscUtilities::checkError;

              const PetscScalar *val;
              xDMPlexPointLocalRead(dm, star[st], fID, dataArray, &val) >> ablate::utilities::PetscUtilities::checkError;

              for (PetscInt d = 0; d < dim; ++d) *div += val[d]*N[d];

              totalVol += ablate::utilities::MathUtilities::MagVector(dim, N);
            }
        }
    }
    DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &nStar, &star) >> ablate::utilities::PetscUtilities::checkError;

    VecRestoreArrayRead(data, &dataArray) >> ablate::utilities::PetscUtilities::checkError;

    if (totalVol > 0.0) *div /= totalVol;

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
//exit(0);
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
//    if ( size==1 ) {
//      throw std::runtime_error("ablate::levelSet::Utilities::VertexUpwindGrad encounted a situation where there are no upwind cells");
//    }
//    if ( size==1 ) {
//      char err[255];
//      sprintf(err, "ablate::levelSet::Utilities::VertexUpwindGrad encounted a situation where there are no upwind cells %f,%f", x0[0], x0[1]);
//      throw std::runtime_error(err);
//    }
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



void SaveVertexData(DM dm, Vec vec, const char fname[255], const ablate::domain::Field *field, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

  ablate::domain::Range range;
  const PetscReal    *array, *val;
  PetscInt      dim = subDomain->GetDimensions();
  MPI_Comm      comm = PetscObjectComm((PetscObject)dm);
  int rank, size;
  MPI_Comm_size(comm, &size) >> ablate::utilities::MpiUtilities::checkError;
  MPI_Comm_rank(comm, &rank) >> ablate::utilities::MpiUtilities::checkError;

  ablate::domain::GetRange(dm, nullptr, 0, range);

  VecGetArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt r = 0; r < size; ++r) {
    if ( rank==r ) {

      FILE *f1;
      if ( rank==0 ) f1 = fopen(fname, "w");
      else f1 = fopen(fname, "a");

      for (PetscInt v = range.start; v < range.end; ++v) {
        PetscInt vert = range.points ? range.points[v] : v;
        PetscScalar *coords;

        DMPlexPointLocalFieldRead(dm, vert, field->id, array, &val) >> ablate::utilities::PetscUtilities::checkError;

        DMPlexVertexGetCoordinates(dm, 1, &vert, &coords);

        for (PetscInt d = 0; d < dim; ++d) {
          fprintf(f1, "%+.16e\t", coords[d]);
        }
        fprintf(f1, "%+.16e\n", *val);

        DMPlexVertexRestoreCoordinates(dm, 1, &vert, &coords);
      }

      fclose(f1);
    }
    MPI_Barrier(comm);
  }


  VecRestoreArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
  ablate::domain::RestoreRange(range);
}

void SaveVertexData(const char fname[255], const ablate::domain::Field *field, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

  Vec           vec = subDomain->GetVec(*field);
  DM            dm  = subDomain->GetFieldDM(*field);
  SaveVertexData(dm, vec, fname, field, subDomain);
}

void SaveCellData(DM dm, const Vec vec, const char fname[255], const PetscInt id, PetscInt Nc, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

  ablate::domain::Range range;
  const PetscScalar *array;
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

        if (ablate::levelSet::Utilities::ValidCell(dm, cell)) {

          PetscReal x0[3];
          DMPlexComputeCellGeometryFVM(dm, cell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;
          for (PetscInt d = 0; d < dim; ++d) {
            fprintf(f1, "%+e\t", x0[d]);
          }

          const PetscScalar *val;
          DMPlexPointLocalFieldRead(dm, cell, id, array, &val) >> ablate::utilities::PetscUtilities::checkError;
          for (PetscInt i = 0; i < Nc; ++i) {
            fprintf(f1, "%+e\t", val[i]);
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

void SaveCellData(DM dm, const Vec vec, const char fname[255], const ablate::domain::Field *field, PetscInt Nc, std::shared_ptr<ablate::domain::SubDomain> subDomain) {
  SaveCellData(dm, vec, fname, field->id, Nc, subDomain);
}


// Inter-processor ghost cells are iterated over, so everything should work fine
static void CutCellLevelSetValues(std::shared_ptr<ablate::domain::SubDomain> subDomain, ablate::domain::Range cellRange, ablate::domain::Range vertRange, ablate::domain::ReverseRange reverseVertRange, const PetscInt *cellMask, DM solDM, Vec solVec, const PetscInt vofID, DM auxDM, Vec auxVec, const PetscInt normalID, const PetscInt lsID) {

  const PetscScalar *solArray = nullptr;
  PetscScalar *auxArray = nullptr;
  PetscInt *lsCount;


  VecGetArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;

  DMGetWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &lsCount) >> ablate::utilities::PetscUtilities::checkError;
  PetscArrayzero(lsCount, vertRange.end - vertRange.start) >> ablate::utilities::PetscUtilities::checkError;
  lsCount -= vertRange.start;

  for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
    PetscInt vert = vertRange.GetPoint(v);
    PetscReal *lsVal = nullptr;
    xDMPlexPointLocalRef(auxDM, vert, lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;
    *lsVal = 0.0;
  }

//int rank;
//MPI_Comm_rank(PETSC_COMM_WORLD, &rank) >> ablate::utilities::MpiUtilities::checkError;

  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

    // Only worry about cut-cells
    if ( cellMask[c]==1 ) {
      PetscInt cell = cellRange.GetPoint(c);

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
        xDMPlexPointLocalRef(auxDM, verts[v], lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;
        *lsVal += lsVertVals[v];

        PetscInt vert_i = reverseVertRange.GetIndex(verts[v]);
        ++lsCount[vert_i];
      }

      DMRestoreWorkArray(auxDM, nv, MPIU_REAL, &lsVertVals) >> ablate::utilities::PetscUtilities::checkError;
      DMPlexCellRestoreVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

    }
  }

  // This is no longer needed
  VecRestoreArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
    if ( lsCount[v] > 0 ) {

      PetscInt vert = vertRange.GetPoint(v);

      PetscReal *lsVal = nullptr;
      xDMPlexPointLocalRef(auxDM, vert, lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;

      *lsVal /= lsCount[v];
    }
  }

  lsCount += vertRange.start;
  DMRestoreWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &lsCount) >> ablate::utilities::PetscUtilities::checkError;

  VecRestoreArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;

  subDomain->UpdateAuxLocalVector();

}



// Solve Eqs. (25) - (26) returning alpha^m in Eq. (27)
// dm - Mesh
// P - center cell
// N - neightbor cell
// cellGrad - Cell center gradient, Eq. (24)
// fc - face centroid
// S - The face outward normal
// alpha - alpha^m in Eq. (27)
static void DMPlexCellGradFromCellFluxLimited_2SharedCells(DM dm, const PetscInt P, const PetscInt N, const PetscScalar *dataArray, PetscInt fID, PetscReal cellGrad[], PetscReal fc[], PetscReal S[], PetscReal *alpha) {

  PetscInt       dim;
  DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;


  PetscReal *Pv;
  xDMPlexPointLocalRead(dm, P, fID, dataArray, &Pv) >> ablate::utilities::PetscUtilities::checkError;

  if (ablate::levelSet::Utilities::ValidCell(dm, N)) {

    PetscReal *Nv;
    xDMPlexPointLocalRead(dm, N, fID, dataArray, &Nv) >> ablate::utilities::PetscUtilities::checkError;


    PetscReal Px[dim], Nx[dim];
    DMPlexComputeCellGeometryFVM(dm, P, NULL, Px, NULL) >> ablate::utilities::PetscUtilities::checkError;
    DMPlexComputeCellGeometryFVM(dm, N, NULL, Nx, NULL) >> ablate::utilities::PetscUtilities::checkError;

    PetscReal dist = 0.0;
    for (PetscInt d = 0; d < dim; ++d) dist += PetscSqr(Nx[d] - Px[d]);
    dist = PetscSqrtReal(dist);

    // Approximation of the derivative between the center cell and the neighbor cell
    PetscReal dadc = (*Nv - *Pv)/dist;


    // Inner product between the cell gradient and the cell-face unit normal
    dist = 0.0;
    for (PetscInt d = 0; d < dim; ++d) dist += PetscSqr(S[d]);
    dist = PetscSqrtReal(dist);

    PetscReal dot = 0.0;
    for (PetscInt d = 0; d < dim; ++d) dot += cellGrad[d]*S[d];
    dot /= dist;

    if ((dot*dadc > 0) && (PetscAbsReal(dot) < PetscAbsReal(dadc))) {
      *alpha = *Nv;
    }
    else if ((dot*dadc > 0) && (PetscAbsReal(dot) > PetscAbsReal(dadc))) {
      *alpha = *Pv;
    }
    else {
      PetscReal dN = 0.0, dP = 0.0;
      for (PetscInt d = 0; d < dim; ++d) {
        dN += PetscSqr(fc[d] - Nx[d]);
        dP += PetscSqr(fc[d] - Px[d]);
      }
      dN = PetscSqrtReal(dN);
      dP = PetscSqrtReal(dP);
      *alpha = ((*Pv)*dP + (*Nv)*dN)/(dP + dN);
    }
  }
  else {
    *alpha = *Pv;
  }



}


// See "Anti-diffusion method for interface steepening in two-phase incompressible flow" by So, Hu, and Adams
void DMPlexCellGradFromCellFluxLimited(DM dm, const PetscInt c, Vec data, PetscInt fID, PetscScalar g[]) {

    PetscInt       dim;
    DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

    const PetscScalar *dataArray;
    VecGetArrayRead(data, &dataArray) >> ablate::utilities::PetscUtilities::checkError;

    PetscReal cellGrad[dim];
    DMPlexCellGradFromCell(dm, c, data, fID, 0, cellGrad) >> ablate::utilities::PetscUtilities::checkError;

    // Get all faces of the cell
    PetscInt       nFaces;
    const PetscInt *faces;
    DMPlexGetConeSize(dm, c, &nFaces) >> ablate::utilities::PetscUtilities::checkError;
    DMPlexGetCone(dm, c, &faces) >> ablate::utilities::PetscUtilities::checkError;

    for (PetscInt d = 0; d < dim; ++d) g[d] = 0.0;

    for (PetscInt f = 0; f < nFaces; ++f) {

      // Compute the face center location and the outward surface area normal
      PetscReal S[dim], fc[dim];
      DMPlexFaceCentroidOutwardAreaNormal(dm, c, faces[f], fc, S) >> ablate::utilities::PetscUtilities::checkError;

      // The cells sharing this face
      PetscInt       nSharedCells;
      const PetscInt *sharedCells;
      DMPlexGetSupportSize(dm, faces[f], &nSharedCells) >> ablate::utilities::PetscUtilities::checkError;
      DMPlexGetSupport(dm, faces[f], &sharedCells) >> ablate::utilities::PetscUtilities::checkError;

      PetscReal limitedVal = 0.0;
      switch (nSharedCells) {
        case 1:
        {
          PetscReal *val;
          xDMPlexPointLocalRead(dm, c, fID, dataArray, &val) >> ablate::utilities::PetscUtilities::checkError;
          limitedVal = *val;
          break;
        }
        case 2:
        {
          PetscInt P = sharedCells[0], N = sharedCells[1];
          if (N==c) {
            P = sharedCells[1];
            N = sharedCells[0];
          }
          DMPlexCellGradFromCellFluxLimited_2SharedCells(dm, P, N, dataArray, fID, cellGrad, fc, S, &limitedVal);
          break;
        }
        default:
        {
          PetscPrintf(PETSC_COMM_WORLD, "Too many cells sharing this face.\n");
          exit(0);
        }
      }


      for (PetscInt d = 0; d < dim; ++d) g[d] += limitedVal*S[d];
    }

    // Center of the cell
    PetscReal cellVolume;
    DMPlexComputeCellGeometryFVM(dm, c, &cellVolume, NULL, NULL) >> ablate::utilities::PetscUtilities::checkError;
    for (PetscInt d = 0; d < dim; ++d) g[d] /= cellVolume;

}





// See "Anti-diffusion method for interface steepening in two-phase incompressible flow" by So, Hu, and Adams
// NOT IDEAL WITH ALL OF THE COPIED CODE
void DMPlexCellDivFromCellFluxLimited(DM dm, const PetscInt c, Vec data, const PetscInt vofID, const PetscInt gradID, const PetscInt cellGradID, const PetscReal *sharpGrad, PetscScalar *div) {

    PetscInt       dim;
    DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

    PetscReal         h;
    DMPlexGetMinRadius(dm, &h) >> ablate::utilities::PetscUtilities::checkError;
    h *= 2.0;
    PetscReal eps = 0.0*0.25*PetscPowReal(h, 0.9);

    const PetscScalar *dataArray;
    VecGetArrayRead(data, &dataArray) >> ablate::utilities::PetscUtilities::checkError;


    // Get all faces of the cell
    PetscInt       nFaces;
    const PetscInt *faces;
    DMPlexGetConeSize(dm, c, &nFaces) >> ablate::utilities::PetscUtilities::checkError;
    DMPlexGetCone(dm, c, &faces) >> ablate::utilities::PetscUtilities::checkError;


    *div = 0.0;

    for (PetscInt f = 0; f < nFaces; ++f) {

      // Compute the face center location and the outward surface area normal
      PetscReal S[dim];
      DMPlexFaceCentroidOutwardAreaNormal(dm, c, faces[f], NULL, S) >> ablate::utilities::PetscUtilities::checkError;

      // The cells sharing this face
      PetscInt       nSharedCells;
      const PetscInt *sharedCells;
      DMPlexGetSupportSize(dm, faces[f], &nSharedCells) >> ablate::utilities::PetscUtilities::checkError;
      DMPlexGetSupport(dm, faces[f], &sharedCells) >> ablate::utilities::PetscUtilities::checkError;

      const PetscReal *cVOF;
      xDMPlexPointLocalRead(dm, c, vofID, dataArray, &cVOF) >> ablate::utilities::PetscUtilities::checkError;
      PetscReal cFlux = (*cVOF)*(1.0 - (*cVOF));

      PetscReal nrm = ablate::utilities::MathUtilities::MagVector(dim, &(sharpGrad[dim*c]));
      PetscReal scale = 0.5*(1.0 - cos(2.0*M_PI*nrm*2.0*h));
//      cFlux *= cScale;


      PetscReal *cn;
      xDMPlexPointLocalRead(dm, c, cellGradID, dataArray, &cn) >> ablate::utilities::PetscUtilities::checkError;

      PetscReal *pn = cn;

      PetscReal pFlux = cFlux;


      if (nSharedCells==2) {

        PetscInt nc = sharedCells[1]; // Neighbor cell
        if (sharedCells[1] == c) nc = sharedCells[0];

        if (ablate::levelSet::Utilities::ValidCell(dm, nc)) {
          PetscReal *pVOF;
          xDMPlexPointLocalRead(dm, nc, vofID, dataArray, &pVOF) >> ablate::utilities::PetscUtilities::checkError;
          pFlux = (*pVOF)*(1.0 - (*pVOF));


          PetscReal cx[3], px[3];
          DMPlexComputeCellGeometryFVM(dm, c, NULL, cx, NULL) >> ablate::utilities::PetscUtilities::checkError;
          DMPlexComputeCellGeometryFVM(dm, nc, NULL, px, NULL) >> ablate::utilities::PetscUtilities::checkError;

          PetscReal dist = 0.0;
          for (PetscInt d = 0; d < dim; ++d) dist += PetscSqr(cx[d] - px[d]);
          dist = PetscSqrtReal(dist);

          nrm = ((*pVOF) - (*cVOF))/dist;
          scale = 0.5*(1.0 - cos(2.0*M_PI*nrm*h));

//          PetscReal pNrm = ablate::utilities::MathUtilities::MagVector(dim, &(sharpGrad[dim*nc]));
//          PetscReal pScale = 0.5*(1.0 - cos(2.0*M_PI*pNrm*2.0*h));
//          pFlux *= pScale;

          xDMPlexPointLocalRead(dm, nc, cellGradID, dataArray, &pn) >> ablate::utilities::PetscUtilities::checkError;


        }
      }
      for (PetscInt d = 0; d < dim; ++d) *div += scale*0.5*(cFlux*cn[d] + pFlux*pn[d])*S[d];
    }

    // Center of the cell
    PetscReal cellVolume;
    DMPlexComputeCellGeometryFVM(dm, c, &cellVolume, NULL, NULL) >> ablate::utilities::PetscUtilities::checkError;
    *div /= cellVolume;

    for (PetscInt d = 0; d < dim; ++d) {
      PetscReal dg[dim];
      DMPlexCellGradFromVertex(dm, c, data, gradID, d, dg);
      *div -= eps*dg[d];
    }

}


// vofDM, vofID - The original VOF data in the SOL vector. Cell-centered
// gradID - Location to store gradient of vof field in the AUX vector. Cell-centered
// sharpID - Sharpened VOF in the AUX vector. Cell-centered

void ablate::levelSet::Utilities::SharpenVOF(std::shared_ptr<ablate::domain::SubDomain> subDomain, ablate::domain::Range cellRange, ablate::domain::Range vertRange, const Vec solVec, const Vec auxVec, DM solDM, DM auxDM, const PetscInt vofID, const PetscInt vertexGradID, const PetscInt cellGradID, const PetscInt sharpID) {

  PetscInt       dim;
  DMGetDimension(auxDM, &dim) >> ablate::utilities::PetscUtilities::checkError;

  const PetscScalar *solArray = nullptr;
  PetscScalar       *auxArray = nullptr;
  VecGetArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
  PetscReal         h;
  DMPlexGetMinRadius(solDM, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0;
//  const PetscReal   dt = 0.1*h*h;
  const PetscReal   dt = 0.25*PetscPowReal(h, 1.1);;

  const PetscReal vofRange[2] = {1.e-5, 1.0-1.e-5};


  // Copy the original VOF to the sharpened one and compute the cell normal
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

    PetscInt cell = cellRange.GetPoint(c);

    if (ablate::levelSet::Utilities::ValidCell(auxDM, cell)) {

      const PetscScalar *vofVal = nullptr;
      xDMPlexPointLocalRead(solDM, cell, vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

      PetscScalar *sharpVal = nullptr;
      xDMPlexPointLocalRef(auxDM, cell, sharpID, auxArray, &sharpVal) >> ablate::utilities::PetscUtilities::checkError;

      *sharpVal = *vofVal;

      PetscScalar *grad = nullptr;
      xDMPlexPointLocalRef(auxDM, cell, cellGradID, auxArray, &grad) >> ablate::utilities::PetscUtilities::checkError;

      if ((*vofVal > vofRange[0]) && (*vofVal < vofRange[1])) {

        DMPlexCellGradFromCell(solDM, cell, solVec, vofID, 0, grad) >> ablate::utilities::PetscUtilities::checkError;

        ablate::utilities::MathUtilities::NormVector(dim, grad);
      }
      else {
        for (PetscInt d = 0; d < dim; ++d) grad[d] = 0.0;
      }

//      if (*sharpVal>vofRange[1]) *sharpVal = 1.0;
//      else if(*sharpVal<vofRange[0]) *sharpVal = 0.0;

    }

  }
  subDomain->UpdateAuxLocalVector();
SaveCellData(auxDM, auxVec, "vof.txt", sharpID, 1, subDomain);



  PetscReal *sharpGrad;
  DMGetWorkArray(auxDM, dim*(cellRange.end - cellRange.start), MPIU_REAL, &sharpGrad) >> ablate::utilities::PetscUtilities::checkError;
  PetscArrayzero(sharpGrad, dim*(cellRange.end - cellRange.start));
  sharpGrad -= dim*cellRange.start; // offset so that we can use start->end


  PetscReal maxDiff = PETSC_MAX_REAL;
  PetscInt iter = 0;
  const PetscReal tol = 0.1*dt;
  while (maxDiff > tol) {
    ++iter;

    // Compute the gradient
    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

      PetscInt vert = vertRange.GetPoint(v);

      bool validVert = true;

      PetscInt nCells, *cells;
      DMPlexVertexGetCells(auxDM, vert, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

      for (PetscInt i = 0; i < nCells; ++i) {
        validVert = validVert && ablate::levelSet::Utilities::ValidCell(auxDM, cells[i]);
      }

      DMPlexVertexRestoreCells(auxDM, vert, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;


      PetscScalar *grad = nullptr;
      xDMPlexPointLocalRef(auxDM, vert, vertexGradID, auxArray, &grad) >> ablate::utilities::PetscUtilities::checkError;

      if (validVert) {
        DMPlexVertexGradFromCell(auxDM, vert, auxVec, sharpID, 0, grad) >> ablate::utilities::PetscUtilities::checkError;

//        PetscReal nrm = ablate::utilities::MathUtilities::MagVector(dim, grad);
//        nrm *= h;
//        PetscReal scale = 0.05 + (1.0-0.05)*(nrm - sin(2.0*M_PI*nrm)/(2.0*M_PI));
//        for (PetscInt d = 0; d < dim; ++d) grad[d] *= scale;

      }
      else {
        for (PetscInt d = 0; d < dim; ++d) grad[d] = 0.0;
      }
    }


    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      PetscInt cell = cellRange.GetPoint(c);
      if (ablate::levelSet::Utilities::ValidCell(auxDM, cell)) {
        DMPlexCellGradFromCell(auxDM, cell, auxVec, sharpID, 0, &(sharpGrad[dim*c])) >> ablate::utilities::PetscUtilities::checkError;
      }
    }



    subDomain->UpdateAuxLocalVector();

    // Divergence and update
    maxDiff = -1;

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

      PetscInt cell = cellRange.GetPoint(c);

      if (ablate::levelSet::Utilities::ValidCell(auxDM, cell)) {

        const PetscScalar *vofVal = nullptr;
        xDMPlexPointLocalRead(solDM, cell, vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

//        if ((*vofVal > vofRange[0]) && (*vofVal < vofRange[1])) {

          PetscReal div;
          DMPlexCellDivFromCellFluxLimited(auxDM ,cell, auxVec, sharpID, vertexGradID, cellGradID, sharpGrad, &div);

          PetscScalar *sharpVal = nullptr;
          xDMPlexPointLocalRef(auxDM, cell, sharpID, auxArray, &sharpVal) >> ablate::utilities::PetscUtilities::checkError;

          PetscScalar newVal = *sharpVal - dt*div;

          newVal = PetscMax(PetscMin(newVal, 1.0), 0.0);

          if (newVal>vofRange[1]) newVal = 1.0;
          else if(newVal<vofRange[0]) newVal = 0.0;

          maxDiff = PetscMax(maxDiff, PetscAbsScalar(*sharpVal - newVal));

          *sharpVal = newVal;

//        }
      }
    }
    subDomain->UpdateAuxLocalVector();

char fname[255];
sprintf(fname, "sharp%02" PetscInt_FMT".txt", iter);
SaveCellData(auxDM, auxVec, fname, sharpID, 1, subDomain);

    PetscPrintf(PETSC_COMM_WORLD, "%2" PetscInt_FMT": %e\t%e\n", iter, maxDiff, tol);

  }
printf("Here\n");
exit(0);
  VecRestoreArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;

  sharpGrad += dim*cellRange.start; // offset so that we can use start->end
  DMRestoreWorkArray(auxDM, dim*(cellRange.end - cellRange.start), MPIU_REAL, &sharpGrad) >> ablate::utilities::PetscUtilities::checkError;


  exit(0);


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
static std::shared_ptr<ablate::domain::rbf::RBF> vertRBF = nullptr;

static std::shared_ptr<ablate::domain::rbf::RBF> cellRBF = nullptr;

// Temporary for the review
//static PetscInt **cellNeighbors = nullptr, *numberNeighbors = nullptr;

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
void ablate::levelSet::Utilities::Reinitialize(std::shared_ptr<ablate::domain::SubDomain> subDomain, const Vec solVec, const ablate::domain::Field *vofField, const PetscInt nLevels, const ablate::domain::Field *lsField, const ablate::domain::Field *vertexNormalField, const ablate::domain::Field *cellNormalField, const ablate::domain::Field *curvField) {

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


  PetscReal         h = 0.0;
  const PetscInt    dim = subDomain->GetDimensions();   // VOF and LS subdomains must have the same dimension. Can't think of a reason they wouldn't.
  PetscInt          *vertMask = nullptr, *cellMask = nullptr;
  DM                solDM = subDomain->GetDM();
  DM                auxDM = subDomain->GetAuxDM();
  Vec               auxVec = subDomain->GetAuxVector();
  const PetscScalar *solArray = nullptr;
  PetscScalar       *auxArray = nullptr;
  const PetscInt    lsID = lsField->id, vofID = vofField->id, cellNormalID = cellNormalField->id;

  DMPlexGetMinRadius(solDM, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size
printf("%+f\n", h);
//exit(0);
  /***********************************************************************************************/
  // THIS IS TEMPORARY AND NEEDS TO BE MOVED TO THE YAML FILE OR SOMETHING ELSE AFTER THE REVIEW
  /***********************************************************************************************/
  if ( vertRBF==nullptr ) {
    PetscInt polyAug = 3;
    bool doesNotHaveDerivatives = false;
    bool doesNotHaveInterpolation = false;
    bool returnNeighborVertices = true;
    vertRBF = std::make_shared<ablate::domain::rbf::PHS>(polyAug, h, doesNotHaveDerivatives, doesNotHaveInterpolation, returnNeighborVertices);

    vertRBF->Setup(subDomain);       // This causes issues (I think)
    vertRBF->Initialize();  //         Initialize
  }

  if ( cellRBF==nullptr ) {
    PetscInt polyAug = 3;
    bool doesNotHaveDerivatives = false;
    bool doesNotHaveInterpolation = false;
    bool returnNeighborVertices = false;
    cellRBF = std::make_shared<ablate::domain::rbf::GA>(polyAug, h, doesNotHaveDerivatives, doesNotHaveInterpolation, returnNeighborVertices);

    cellRBF->Setup(subDomain);       // This causes issues (I think)
    cellRBF->Initialize();  //         Initialize
  }

  VecGetArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;


  ablate::domain::Range cellRange, vertRange;
  subDomain->GetCellRange(nullptr, cellRange);
  subDomain->GetRange(nullptr, 0, vertRange);

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

SaveCellData(solDM, solVec, "vof.txt", vofField, 1, subDomain);

/**************** Determine the cut-cells and initial unit normal *************************************/

  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

    PetscInt cell = cellRange.GetPoint(c);

    if (ablate::levelSet::Utilities::ValidCell(solDM, cell)) {

      const PetscScalar *vofVal = nullptr;
      xDMPlexPointLocalRead(solDM, cell, vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

      PetscScalar *n = nullptr;
      xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &n);
      for (PetscInt d = 0; d < dim; ++d ) n[d] = 0.0;

      if ( ((*vofVal) > 0.0001) && ((*vofVal) < 0.9999) ) {

        cellMask[c] = 1;    // Mark as a cut-cell

        // Will this crap near the edges of a processor?
        if ( dim > 0 ) n[0] = cellRBF->EvalDer(solDM, solVec, vofID, cell, 1, 0, 0);
        if ( dim > 1 ) n[1] = cellRBF->EvalDer(solDM, solVec, vofID, cell, 0, 1, 0);
        if ( dim > 2 ) n[2] = cellRBF->EvalDer(solDM, solVec, vofID, cell, 0, 0, 1);

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

  subDomain->UpdateAuxLocalVector();



/**************** Iterate to get the level-set values at vertices *************************************/

  // Temporary level-set work array to store old values
  PetscScalar *tempLS;
  DMGetWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_SCALAR, &tempLS) >> ablate::utilities::PetscUtilities::checkError;
  tempLS -= vertRange.start;

  PetscReal maxDiff = 1.0;
  PetscInt iter = 0;

  MPI_Comm auxCOMM = PetscObjectComm((PetscObject)auxDM);

//SaveCellData(auxDM, auxVec, "normal0.txt", cellNormalField, dim, subDomain);

  PetscReal cRange[2] = {PETSC_MAX_REAL, -PETSC_MAX_REAL};
  while ( maxDiff > 1e-3*h && iter<500 ) {

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
    CutCellLevelSetValues(subDomain, cellRange, vertRange, reverseVertRange, cellMask, solDM, solVec, vofID, auxDM, auxVec, cellNormalID, lsID);

    //     Update the normals
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      if (cellMask[c] == 1) {
        PetscInt cell = cellRange.GetPoint(c);
        PetscScalar *n = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &n);
        DMPlexCellGradFromVertex(auxDM, cell, auxVec, lsID, 0, n) >> ablate::utilities::PetscUtilities::checkError;
        ablate::utilities::MathUtilities::NormVector(dim, n);
      }
    }

    subDomain->UpdateAuxLocalVector();


    // Now compute the difference on this processor
    maxDiff = -1.0;
    cRange[0] = PETSC_MAX_REAL;
    cRange[1] = -PETSC_MAX_REAL;
    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

      if (vertMask[v] == 1) {
        PetscInt vert = vertRange.GetPoint(v);
        const PetscReal *newLS = nullptr;
        xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &newLS) >> ablate::utilities::PetscUtilities::checkError;

        maxDiff = PetscMax(maxDiff, PetscAbsReal(tempLS[v] - *newLS));

        cRange[0] = PetscMin(cRange[0], *newLS);
        cRange[1] = PetscMax(cRange[1], *newLS);

      }
    }
    // Get the maximum change across all processors. This also acts as a sync point
    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, auxCOMM);

    PetscPrintf(PETSC_COMM_WORLD, "Cut Cells %" PetscInt_FMT": %+e\n", iter, maxDiff) >> ablate::utilities::PetscUtilities::checkError;
  }

  if (maxDiff > 1e-3*h) {
    SaveCellData(auxDM, auxVec, "normalERROR.txt", cellNormalField, dim, subDomain);
    SaveVertexData(auxDM, auxVec, "ls0ERROR.txt", lsField, subDomain);
    throw std::runtime_error("Interface reconstruction has failed. Please check the number of cut-cells.\n");
  }

  cRange[0] *= -1.0;
  MPI_Allreduce(MPI_IN_PLACE, cRange, 2, MPIU_REAL, MPIU_MAX, auxCOMM);
  cRange[0] *= -1.0;



/**************** Set the data in the rest of the domain to be a large value *************************************/
PetscPrintf(PETSC_COMM_WORLD, "Setting data\n");
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


SaveVertexData(auxDM, auxVec, "ls0.txt", lsField, subDomain);


/**************** Mark the cells that need to be udpated via the reinitialization equation *************************************/
PetscPrintf(PETSC_COMM_WORLD, "Marking cells\n");
  // Mark all of the cells neighboring cells level-by-level.
  // Note that DMPlexGetNeighbors has an issue in parallel whereby cells will be missed due to the unknown partitioning -- Need to investigate
  Vec workVec, workVecGlobal;
  PetscScalar *workArray = nullptr;
  DMGetLocalVector(auxDM, &workVec);
  DMGetGlobalVector(auxDM, &workVecGlobal);

  VecZeroEntries(workVec);

  VecGetArray(workVec, &workArray);
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);
    PetscScalar *maskVal = nullptr;
    xDMPlexPointLocalRef(auxDM, cell, vofID, workArray, &maskVal) >> ablate::utilities::PetscUtilities::checkError;
    *maskVal = cellMask[c];
  }
  VecRestoreArray(workVec, &workArray);

  DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
  DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;

int rank;
MPI_Comm_rank(PETSC_COMM_WORLD, &rank) >> ablate::utilities::MpiUtilities::checkError;
  for (PetscInt l = 1; l <= nLevels; ++l) {

    VecGetArray(workVec, &workArray);
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      PetscInt cell = cellRange.GetPoint(c);
      PetscScalar *maskVal = nullptr;
      xDMPlexPointLocalRef(auxDM, cell, vofID, workArray, &maskVal) >> ablate::utilities::PetscUtilities::checkError;

      if ( PetscAbsScalar(*maskVal - l) < 0.1 ) {
        PetscInt nCells, *cells;
        DMPlexGetNeighbors(solDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
        for (PetscInt i = 0; i < nCells; ++i) {
          PetscScalar *neighborMaskVal = nullptr;
          xDMPlexPointLocalRef(auxDM, cells[i], vofID, workArray, &neighborMaskVal) >> ablate::utilities::PetscUtilities::checkError;
          if ( *neighborMaskVal < 0.5 ) {
            *neighborMaskVal = l + 1;

            cellMask[reverseCellRange.GetIndex(cells[i])] = l + 1;

            PetscScalar *vofVal;
            xDMPlexPointLocalRead(solDM, cells[i], vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

            PetscInt nv, *verts;
            DMPlexCellGetVertices(auxDM, cells[i], &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

            for (PetscInt v = 0; v < nv; ++v) {
              PetscInt id = reverseVertRange.GetIndex(verts[v]);

              if (vertMask[id]==0) {
                vertMask[id] = l + 1;

                PetscScalar *lsVal;
                xDMPlexPointLocalRef(auxDM, verts[v], lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;

                PetscReal sgn = (*vofVal < 0.5 ? +1.0 : -1.0);
                *lsVal = sgn*l*h;
              }
            }
          }
        }
        DMPlexRestoreNeighbors(solDM, cell, 1, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
      }
    }
    VecRestoreArray(workVec, &workArray);

    DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;
  }

  VecRestoreArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;

  subDomain->UpdateAuxLocalVector();
SaveVertexData(auxDM, auxVec, "ls1.txt", lsField, subDomain);


PetscPrintf(PETSC_COMM_WORLD, "Reinit\n");
/**************** Level-set reinitialization equation *************************************/
  const PetscInt vertexNormalID = vertexNormalField->id;
  const PetscInt curvID = curvField->id;



  maxDiff = 1.0;
  iter = 0;


PetscInt *divMask;
DMGetWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &divMask) >> ablate::utilities::PetscUtilities::checkError;
PetscArrayzero(divMask, vertRange.end - vertRange.start) >> ablate::utilities::PetscUtilities::checkError;
divMask -= vertRange.start; // offset so that we can use start->end

//PetscReal *vertGrad;
//DMGetWorkArray(auxDM, dim*(vertRange.end - vertRange.start), MPIU_REAL, &vertGrad) >> ablate::utilities::PetscUtilities::checkError;
//PetscArrayzero(vertGrad, dim*(vertRange.end - vertRange.start)) >> ablate::utilities::PetscUtilities::checkError;
//vertGrad -= dim*vertRange.start; // offset so that we can use start->end

for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

  if (vertMask[v] > 1) {
    PetscInt vert = vertRange.GetPoint(v);
    PetscInt nCells, *cells;
    DMPlexVertexGetCells(auxDM, vert, &nCells, &cells);
    divMask[v] = 1;

    for (PetscInt c = 0; c < nCells; ++c) {
      divMask[v] = divMask[v] && (cellMask[reverseCellRange.GetIndex(cells[c])] > 0);
    }

    DMPlexVertexRestoreCells(auxDM, vert, &nCells, &cells);

  }
}


  while (maxDiff>1.e-3*h && iter<100) {
    ++iter;


    // Determine the current gradient at cells that need updating
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      if (cellMask[c] > 0) {
        PetscInt cell = cellRange.GetPoint(c);
        PetscReal *g = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &g) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexCellGradFromVertex(auxDM, cell, auxVec, lsID, 0, g) >> ablate::utilities::PetscUtilities::checkError;

        PetscReal *nrm = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, curvID, auxArray, &nrm) >> ablate::utilities::PetscUtilities::checkError;
        *nrm = ablate::utilities::MathUtilities::MagVector(dim, g);



      }
    }
    subDomain->UpdateAuxLocalVector();

    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
      if (vertMask[v] > 1) {
        PetscInt vert = vertRange.GetPoint(v);

        PetscReal *g = nullptr;
        xDMPlexPointLocalRef(auxDM, vert, vertexNormalID, auxArray, &g) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexVertexGradFromVertex(auxDM, vert, auxVec, lsID, 0, g) >> ablate::utilities::PetscUtilities::checkError;

      }
    }

    maxDiff = -PETSC_MAX_REAL;

    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

      if (vertMask[v] > 1) {
        PetscInt vert = vertRange.GetPoint(v);

        PetscReal *phi = nullptr;
        xDMPlexPointLocalRef(auxDM, vert, lsID, auxArray, &phi) >> ablate::utilities::PetscUtilities::checkError;

        const PetscReal phi0 = *phi;

        const PetscReal *arrayG = nullptr;
        xDMPlexPointLocalRef(auxDM, vert, vertexNormalID, auxArray, &arrayG) >> ablate::utilities::PetscUtilities::checkError;

        PetscReal g[dim], n[dim];
        PetscReal nrm = ablate::utilities::MathUtilities::MagVector(dim, arrayG);
        for (PetscInt d = 0; d < dim; ++d){
          g[d] = arrayG[d];
          n[d] = arrayG[d]/nrm;
        }

        VertexUpwindGrad(auxDM, auxArray, cellNormalID, vert, PetscSignReal(*phi), g);

        nrm = ablate::utilities::MathUtilities::MagVector(dim, g);

        const PetscReal alphaH = PetscMax(nrm, 1.0)*h;

        PetscReal sgn;
        if (phi0 < -alphaH) sgn = -1.0;
        else if (phi0 > alphaH) sgn = 1.0;
        else sgn = phi0/alphaH + PetscSinReal(M_PI*phi0/alphaH)/M_PI;

        PetscReal gNRM[dim];
        DMPlexVertexGradFromCell(auxDM, vert, auxVec, curvID, 0, gNRM);

        PetscReal dtProd = 0.0;
        if (divMask[v]==1) dtProd = ablate::utilities::MathUtilities::DotVector(dim, gNRM, n);

        *phi += h*(0.0*h*dtProd - sgn*(nrm - 1.0));

        maxDiff = PetscMax(maxDiff, PetscAbsReal(*phi - phi0));

      }
    }

     // Get the maximum change across all processors. This also acts as a sync point
    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, auxCOMM);

    subDomain->UpdateAuxLocalVector();

    PetscPrintf(PETSC_COMM_WORLD, "Reinit %3" PetscInt_FMT": %e\n", iter, maxDiff);

  }


divMask += vertRange.start; // offset so that we can use start->end
DMRestoreWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &divMask) >> ablate::utilities::PetscUtilities::checkError;

SaveVertexData(auxDM, auxVec, "ls2.txt", lsField, subDomain);
printf("1617\n");


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

  subDomain->UpdateAuxLocalVector();

  // Curvature
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

    PetscInt cell = cellRange.GetPoint(c);
    PetscScalar *H = nullptr;
    xDMPlexPointLocalRef(auxDM, cell, curvID, auxArray, &H);

    if (cellMask[c] >0 ) {


//      *H = ablate::levelSet::geometry::Curvature(vertRBF, lsField, cell);

//      *H = PetscMax(*H, -1.0/h);
//      *H = PetscMin(*H,  1.0/h);

      *H = 0;
      for (PetscInt d = 0; d < dim; ++d) {
        PetscReal g[dim];
//        DMPlexCellGradFromCell(auxDM, cell, auxVec, cellNormalID, d, g) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexCellGradFromVertex(auxDM, cell, auxVec, vertexNormalID, d, g) >> ablate::utilities::PetscUtilities::checkError;
        *H += g[d];
      }
    }
    else {
      *H = 0.0;
    }

  }

  subDomain->UpdateAuxLocalVector();

SaveCellData(auxDM, auxVec, "curv0.txt", curvField, 1, subDomain);

  // Extension
  PetscInt vertexCurvID = lsID; // Store the vertex curvatures in the work vec at the same location as the level-set



#if 0 // Smoothing iteration
  for (PetscInt iter = 0; iter < 10; ++iter) {

    VecGetArray(workVec, &workArray);
    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

      PetscInt vert = vertRange.GetPoint(v);
      PetscReal *vertexH = nullptr;
      xDMPlexPointLocalRef(auxDM, vert, vertexCurvID, workArray, &vertexH);

      if (vertMask[v] > 0) {

  //      for (PetscInt d = 0; d < dim; ++d) {
  //        PetscReal g[dim];
  //        DMPlexVertexGradFromCell(auxDM, vert, auxVec, cellNormalID, d, g);
  //        *vertexH += g[d];
  //      }


          PetscInt nc = 0;
          *vertexH = 0.0;

          PetscInt nCells, *cells;
          DMPlexVertexGetCells(auxDM, vert, &nCells, &cells);
          for (PetscInt c = 0; c < nCells; ++c) {

            PetscInt id = reverseCellRange.GetIndex(cells[c]);

            if (cellMask[id] > 0) {
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
        *vertexH = 0.0;
      }
    }
    VecRestoreArray(workVec, &workArray);

    DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;

    VecGetArray(workVec, &workArray);
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

      PetscInt cell = cellRange.GetPoint(c);

      if (cellMask[c] > 0) {

        PetscScalar *H = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, curvID, auxArray, &H);

        PetscInt nVerts, *verts;
        DMPlexCellGetVertices(auxDM, cell, &nVerts, &verts) >> ablate::utilities::PetscUtilities::checkError;

        *H = 0.0;

        PetscInt nv = 0;
        for (PetscInt v = 0; v < nVerts; ++v) {

          PetscInt id = reverseVertRange.GetIndex(verts[v]);

          if (vertMask[id] > 0) {

            PetscScalar *vH = nullptr;
            xDMPlexPointLocalRef(auxDM, verts[v], vertexCurvID, workArray, &vH);

            *H += *vH;
            ++nv;
          }

        }
        *H /= nv;


        DMPlexCellRestoreVertices(auxDM, cell, &nVerts, &verts) >> ablate::utilities::PetscUtilities::checkError;
      }
    }
    VecRestoreArray(workVec, &workArray);

    subDomain->UpdateAuxLocalVector();

  } // End smoothing iteration
#endif


  DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
  DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;


  SaveVertexData(auxDM, workVec, "vertexH0.txt", lsField, subDomain);
//printf("1787\n");
//exit(0);

  maxDiff = PETSC_MAX_REAL;
  iter = 0;
  while ( maxDiff>5e-2 && iter<500) {
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

    PetscReal oldMaxDiff = maxDiff;

    maxDiff = -PETSC_MAX_REAL;

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

        *H -= 0.5*h*s*dH;
//        *H = PetscMax(*H, -1.0/h);
//        *H = PetscMin(*H,  1.0/h);


        PetscReal *mag = nullptr;
        xDMPlexPointLocalRef(auxDM, vert, vertexNormalID, workArray, &mag) >> ablate::utilities::PetscUtilities::checkError;
        *mag = PetscAbsReal(dH);
      }
    }
    VecRestoreArray(workVec, &workArray);

    DMLocalToGlobal(auxDM, workVec, INSERT_VALUES, workVecGlobal) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(auxDM, workVecGlobal, INSERT_VALUES, workVec) >> utilities::PetscUtilities::checkError;



    // This is temporary until after the review.
    // The norm magnitude is incorrect at the edge of processor domains. There needs to be a way to identify
    //  cell which are ghost cells as they will have incorrect answers.
    VecGetArray(workVec, &workArray);
    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
      if (vertMask[v] > 0) {
        PetscInt vert = vertRange.GetPoint(v);
        const PetscReal *mag = nullptr;
        xDMPlexPointLocalRead(auxDM, vert, vertexNormalID, workArray, &mag) >> ablate::utilities::PetscUtilities::checkError;
        maxDiff = PetscMax(maxDiff, PetscAbsReal(*mag));
      }
    }

    VecRestoreArray(workVec, &workArray);


     // Get the maximum change across all processors. This also acts as a sync point
    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, auxCOMM);

    PetscPrintf(PETSC_COMM_WORLD, "Extension %3" PetscInt_FMT": %e\n", iter, maxDiff);


    if ((maxDiff > oldMaxDiff) && (maxDiff<1e-1)) iter = PETSC_INT_MAX;


  }

  SaveVertexData(auxDM, workVec, "vertexCurv.txt", lsField, subDomain);


  // Now set the curvature at the cell-center via averaging
  VecGetArray(workVec, &workArray);
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    if (cellMask[c] > 0) {
      PetscInt cell = cellRange.GetPoint(c);

//      PetscScalar *n = nullptr;
//      xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &n);
//      if ( dim > 0 ) n[0] = rbf->EvalDer(auxDM, auxVec, lsID, cell, 1, 0, 0);
//      if ( dim > 1 ) n[1] = rbf->EvalDer(auxDM, auxVec, lsID, cell, 0, 1, 0);
//      if ( dim > 2 ) n = rbf->EvalDer(auxDM, auxVec, lsID, cell, 0, 0, 1);
//      ablate::utilities::MathUtilities::NormVector(dim, n);


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
//      *cellH = PetscMax(*cellH, -1.0/h);
//      *cellH = PetscMin(*cellH,  1.0/h);

//*cellH = 5.0*tanh(*cellH/5.0);

//*cellH = 1.0;

      DMPlexCellRestoreVertices(auxDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
    }
  }
  VecRestoreArray(workVec, &workArray);

  subDomain->UpdateAuxLocalVector();


  SaveCellData(auxDM, auxVec, "H.txt", curvField, 1, subDomain);
printf("1960\n");
exit(0);
//  SaveVertexData(auxDM, workVec, "vertexH1.txt", lsField, subDomain);
  DMRestoreLocalVector(auxDM, &workVec) >> utilities::PetscUtilities::checkError;
  DMRestoreGlobalVector(auxDM, &workVecGlobal) >> utilities::PetscUtilities::checkError;



  // Cleanup all memory
//  closestCell += vertRange.start; // offset so that we can use start->end
//  DMRestoreWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &closestCell) >> ablate::utilities::PetscUtilities::checkError;
  tempLS += vertRange.start;
  DMRestoreWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_SCALAR, &tempLS) >> ablate::utilities::PetscUtilities::checkError;
  vertMask += vertRange.start; // Reset the offset, otherwise DMRestoreWorkArray will return unexpected results
  DMRestoreWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &vertMask) >> ablate::utilities::PetscUtilities::checkError;
  cellMask += cellRange.start; // Reset the offset, otherwise DMRestoreWorkArray will return unexpected results
  DMRestoreWorkArray(solDM, cellRange.end - cellRange.start, MPIU_INT, &cellMask) >> ablate::utilities::PetscUtilities::checkError;

  subDomain->RestoreRange(vertRange);

  VecRestoreArray(auxVec, &auxArray) >> utilities::PetscUtilities::checkError;


}
