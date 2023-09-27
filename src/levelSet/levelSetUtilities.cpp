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


//// There should be an Internal function which does the common parts of the two VertexUpwindGrad calls to remove duplication
//void ablate::levelSet::Utilities::VertexUpwindGrad(DM dm, Vec vec, const PetscInt fid, const PetscInt p, const PetscReal direction, PetscReal *g) {
//  // Given a field determine the gradient at a vertex by doing a weighted average of the surrounding cell-centered gradients.
//  // The upwind direction is determined using the dot product between the cell-centered gradient and the vector connecting the cell-center
//  //    and the vertex

//  PetscInt          dim;
//  PetscReal         totalVol = 0.0;
//  PetscScalar       *coords;
//  const PetscScalar *array;

//  DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

//  // Obtain all cells which use this vertex
//  PetscInt nCells, *cells;
//  DMPlexVertexGetCells(dm, p, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

//  DMPlexVertexGetCoordinates(dm, 1, &p, &coords);

//  VecGetArrayRead(vec, &array);

//  for (PetscInt d = 0; d < dim; ++d) {
//    g[d] = 0.0;
//  }

//  for (PetscInt c = 0; c < nCells; ++c) {

//    PetscReal x0[3];
//    DMPlexComputeCellGeometryFVM(dm, cells[c], NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

//    PetscScalar *f;
//    xDMPlexPointLocalRead(dm, cells[c], fid, array, &f) >> utilities::PetscUtilities::checkError;


//    PetscScalar dot = 0.0;
//    for (PetscInt d = 0; d < dim; ++d) {
//      dot += f[d]*(coords[d] - x0[d]);
//    }

//    dot *= direction;

//    if (dot>=0.0) {

////      PetscReal vol;
////      DMPlexComputeCellGeometryFVM(dm, cells[c], &vol, NULL, NULL) >> ablate::utilities::PetscUtilities::checkError;
//      totalVol += dot;

//      // Weighted average of the surrounding cell-center gradients.
//      //  Note that technically this is (in 2D) the area of the quadrilateral that is formed by connecting
//      //  the vertex, center of the neighboring edges, and the center of the triangle. As the three quadrilaterals
//      //  that are formed this way all have the same area, there is no need to take into account the 1/3. Something
//      //  similar should hold in 3D and for other cell types that ABLATE uses.
//      for (PetscInt d = 0; d < dim; ++d) {
//        g[d] += dot*f[d];
//      }
//    }

//  }
//  DMPlexVertexRestoreCoordinates(dm, 1, &p, &coords);
//  DMPlexVertexRestoreCells(dm, p, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
//  VecRestoreArrayRead(vec, &array) >> utilities::PetscUtilities::checkError;

//  // Error checking
//  if ( PetscAbs(totalVol) < ablate::utilities::Constants::small ) {
//    throw std::runtime_error("ablate::levelSet::Utilities::VertexUpwindGrad encounted a situation where there are no upwind cells.");
//  }

//  for (PetscInt d = 0; d < dim; ++d) {
//    g[d] /= totalVol;
//  }
//}




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
  PetscScalar       coords[3], n[3];

  DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

  ablate::utilities::MathUtilities::NormVector(dim, g, n);

  DMPlexComputeCellGeometryFVM(dm, v, NULL, coords, NULL) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt d = 0; d < dim; ++d) {
    g[d] = 0.0;
  }


  // Obtain all cells which use this vertex
  PetscInt nCells, *cells;
  DMPlexVertexGetCells(dm, v, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;



  for (PetscInt c = 0; c < nCells; ++c) {
    PetscReal x0[3];
    DMPlexComputeCellGeometryFVM(dm, cells[c], NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

    PetscInt id = cells[c];
    AOApplicationToPetsc(cellToIndex, 1, &id);

    if (id>-1) {  // If id==-1 then the cell is does not have any values in gradArray. Ignore it.

      PetscReal dot = 0.0;
      for (PetscInt d = 0; d < dim; ++d) {
        dot += n[d]*(coords[d] - x0[d]);
      }

      dot *= direction;

      if (dot>=0.0) {

      const PetscScalar *cellGrad;
      xDMPlexPointLocalRead(dm, cells[c], gradID, gradArray, &cellGrad) >> ablate::utilities::PetscUtilities::checkError;

        // Weighted average of the surrounding cell-center gradients.
        //  Note that technically this is (in 2D) the area of the quadrilateral that is formed by connecting
        //  the vertex, center of the neighboring edges, and the center of the triangle. As the three quadrilaterals
        //  that are formed this way all have the same area, there is no need to take into account the 1/3. Something
        //  similar should hold in 3D and for other cell types that ABLATE uses.
        for (PetscInt d = 0; d < dim; ++d) {
          g[d] += dot*gradArray[id*dim + d];
        }
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

    if (c0) {
      *c0 = centerVal;
    }



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
        const PetscReal *val;
        xDMPlexPointLocalRead(dm, verts[i], lsField->id, array, &val) >> ablate::utilities::PetscUtilities::checkError;
        c[i] = *val;
    }
    VecRestoreArrayRead(vec, &array) >> ablate::utilities::PetscUtilities::checkError;

    ablate::levelSet::Utilities::VOF(dm, cell, c, vof, area, vol);

    DMRestoreWorkArray(dm, nv, MPI_REAL, &c);
    DMPlexCellRestoreVertices(dm, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
}



static void SaveVertexData(DM dm, Vec vec, const char fname[255], const ablate::domain::Field *field, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

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

static void SaveVertexData(const char fname[255], const ablate::domain::Field *field, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

  Vec           vec = subDomain->GetVec(*field);
  DM            dm  = subDomain->GetFieldDM(*field);
  SaveVertexData(dm, vec, fname, field, subDomain);
}


void SaveCellData(const char fname[255], const ablate::domain::Field *field, PetscInt Nc, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

  ablate::domain::Range range;
  PetscReal    *array, *val;
  Vec           vec = subDomain->GetVec(*field);
  DM            dm  = subDomain->GetFieldDM(*field);
  PetscInt      dim = subDomain->GetDimensions();
  MPI_Comm      comm = PetscObjectComm((PetscObject)dm);
  int rank, size;
  MPI_Comm_size(comm, &size) >> ablate::utilities::MpiUtilities::checkError;
  MPI_Comm_rank(comm, &rank) >> ablate::utilities::MpiUtilities::checkError;





  for (PetscInt r = 0; r < size; ++r) {
    if ( rank==r ) {

      FILE *f1;
      if ( rank==0 ) f1 = fopen(fname, "w");
      else f1 = fopen(fname, "a");

      for (PetscInt c = range.start; c < range.end; ++c) {
        PetscInt cell = range.points ? range.points[c] : c;

        PetscReal x0[3];
        DMPlexComputeCellGeometryFVM(dm, cell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexPointLocalFieldRef(dm, cell, field->id, array, &val) >> ablate::utilities::PetscUtilities::checkError;

        for (PetscInt d = 0; d < dim; ++d) {
          fprintf(f1, "%+f\t", x0[d]);
        }

        for (PetscInt i = 0; i < Nc; ++i) {
          fprintf(f1, "%+f\t", val[i]);
        }
        fprintf(f1, "\n");
      }
      fclose(f1);
    }

    MPI_Barrier(PETSC_COMM_WORLD);
  }


  VecRestoreArray(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
  ablate::domain::RestoreRange(range);
}

// Handle the case where the signs are different
void checkSigns(PetscReal existingVal, PetscReal newVal) {
  if (existingVal != PETSC_MAX_REAL && ((existingVal > 0 && newVal < 0) || (existingVal < 0 && newVal > 0))) {
    PetscPrintf(PETSC_COMM_SELF, "Error: Different signs for shared vertex.\n");
    }
}


// vofField - Field containing the cell volume-of-fluid
// cellNormalField - Unit normals at cell-centers. This is pre-computed and an input
// lsField - The updated level-set values at vertices
static void CutCellLevelSetValues(std::shared_ptr<ablate::domain::SubDomain> subDomain, ablate::domain::Range cellRange, ablate::domain::Range vertRange, ablate::domain::ReverseRange reverseVertRange, PetscInt *cellMask, const ablate::domain::Field *vofField, const ablate::domain::Field *cellNormalField, const ablate::domain::Field *lsField) {

  DM              solDM = subDomain->GetDM(), auxDM = subDomain->GetAuxDM();
  Vec             solVec = subDomain->GetSolutionVector(), auxVec = subDomain->GetAuxVector();
  const PetscInt  vofID = vofField->id;
  const PetscInt  normalID = cellNormalField->id;
  const PetscInt  lsID = lsField->id;

  const PetscScalar *solArray;
  PetscScalar *auxArray;

  VecGetArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;
  VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;

  PetscInt *lsCount;
  DMGetWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &lsCount) >> ablate::utilities::PetscUtilities::checkError;
  PetscArrayzero(lsCount, vertRange.end - vertRange.start) >> ablate::utilities::PetscUtilities::checkError;
  lsCount -= vertRange.start;

  for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
    PetscInt vert = vertRange.GetPoint(v);
    PetscReal *lsVal;
    xDMPlexPointLocalRef(auxDM, vert, lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;
    *lsVal = 0.0;
  }


  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {

    // Only worry about cut-cells
    if ( cellMask[c] == 1 ) {

      PetscInt cell = cellRange.GetPoint(c);

      // The VOF for the cell
      const PetscScalar *vofVal;
      xDMPlexPointLocalRead(solDM, cell, vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

      // The pre-computed cell-centered normal
      const PetscScalar *n;
      xDMPlexPointLocalRead(auxDM, cell, normalID, auxArray, &n) >> ablate::utilities::PetscUtilities::checkError;

      PetscInt nv, *verts;
      DMPlexCellGetVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

      PetscReal *lsVertVals = NULL;
      DMGetWorkArray(auxDM, nv, MPIU_REAL, &lsVertVals) >> ablate::utilities::PetscUtilities::checkError;

      // Level set values at the vertices
      ablate::levelSet::Utilities::VertexLevelSet_VOF(auxDM, cell, *vofVal, n, &lsVertVals, NULL);

      for (PetscInt v = 0; v < nv; ++v) {
        PetscScalar *lsVal;

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
      PetscReal *lsVal;
      xDMPlexPointLocalRef(auxDM, vert, lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;

      *lsVal /= lsCount[v];
    }
  }

  VecRestoreArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
  lsCount += vertRange.start;
  DMRestoreWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &lsCount) >> ablate::utilities::PetscUtilities::checkError;

  subDomain->UpdateAuxLocalVector();


}


// Compute the level-set field that corresponds to a given VOF field
// The steps are:
//  1 - Determine the level-set field in cells containing a VOF value between 0 and 1
//  2 - Mark the required number of vertices (based on the cells) next to the interface cells
//  3 - Iterate over vertices EXCEPT for those with cut-cells until converged
//  4 - We may want to look at a fourth step which improve the accuracy
void ablate::levelSet::Utilities::Reinitialize(std::shared_ptr<ablate::domain::SubDomain> subDomain, const ablate::domain::Field *vofField, const PetscInt nLevels, const ablate::domain::Field *lsField) {

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
  Vec               solVec = subDomain->GetSolutionVector();
  Vec               auxVec = subDomain->GetAuxVector();
  const PetscScalar *solArray = nullptr;
  PetscScalar       *auxArray = nullptr;
  const PetscInt    lsID = lsField->id, vofID = vofField->id, cellNormalID = cellNormalField->id;

  DMPlexGetMinRadius(solDM, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size

  SaveCellData("vof.txt", vofField, 1, subDomain);

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
  for (PetscInt i = vertRange.start; i < vertRange.end; ++i) {
    vertMask[i] = -1; //  Ignore the vertex
  }
  for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
    cellMask[i] = -1; // Ignore the cell
  }


  PetscInt *closestCell = nullptr;
  DMGetWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_INT, &closestCell) >> ablate::utilities::PetscUtilities::checkError;
  PetscArrayzero(closestCell, vertRange.end - vertRange.start);
  closestCell -= vertRange.start; // offset so that we can use start->end

  // Setup the cut-cell locations and the initial unit normal estimate
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);

    const PetscScalar *vofVal;
    xDMPlexPointLocalRead(solDM, cell, vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

    // Set the initial normal vector equal to zero. This will ensure that any cells downwind during the PDE update won't have a contribution
    PetscScalar *n;
    xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &n);
    for (PetscInt d = 0; d < dim; ++d ) n[d] = 0.0;

    // Only worry about cut-cells
    if ( ((*vofVal) > ablate::utilities::Constants::small) && ((*vofVal) < (1.0 - ablate::utilities::Constants::small)) ) {

      cellMask[c] = 0;  // Mark as a cut-cell

      // Compute an estimate of the unit normal at the cell-center
      DMPlexCellGradFromCell(solDM, cell, solVec, vofID, 0, n);
      ablate::utilities::MathUtilities::NormVector(dim, n);
      for (PetscInt d = 0; d < dim; ++d) n[d] *= -1.0;

      // Mark all vertices of this cell as associated with a cut-cell
      PetscInt nv, *verts;
      DMPlexCellGetVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
      for (PetscInt v = 0; v < nv; ++v) {
        // Mark as a cut-cell vertex
        vertMask[reverseVertRange.GetIndex(verts[v])] = 0; // Mark vertex as associated with a cut-cell
      }
      DMPlexCellRestoreVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
    }
  }

  // Temporary level-set work array to store old or new values, as appropriate
  PetscScalar *tempLS;
  DMGetWorkArray(auxDM, vertRange.end - vertRange.start, MPIU_SCALAR, &tempLS) >> ablate::utilities::PetscUtilities::checkError;
  tempLS -= vertRange.start;

  PetscReal maxDiff = 1.0;
  PetscInt iter = 0;

  MPI_Comm auxCOMM = PetscObjectComm((PetscObject)auxDM);


  while ( maxDiff > 1e-3*h && iter<8 ) {

    ++iter;

      PetscReal x0[3];
      DMPlexComputeCellGeometryFVM(lsDM, vert, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

        const PetscReal *oldLS;
        xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &oldLS) >> ablate::utilities::PetscUtilities::checkError;
        tempLS[v] = *oldLS;
      }
    }

    // This updates the lsField by taking the average vertex values necessary to match the VOF in cutcells
    CutCellLevelSetValues(subDomain, cellRange, vertRange, reverseVertRange, cellMask, vofField, cellNormalField, lsField);

    // Now compute the difference on this processor
    maxDiff = -1.0;
    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

      *lsVal = 0.0;
      PetscReal totalWeight = 0.0;
      for (PetscInt i = 0; i < nCells; ++i) {
        PetscScalar *vofVal, x[3];
        xDMPlexPointLocalRead(vofDM, cells[i], vofID, vofArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexComputeCellGeometryFVM(vofDM, cells[i], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;

        const PetscReal *newLS;
        xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &newLS) >> ablate::utilities::PetscUtilities::checkError;

        maxDiff = PetscMax(maxDiff, PetscAbsReal(tempLS[v] - *newLS));

      }
      *lsVal /= totalWeight;

      *lsVal = -2.0*(2.0*(*lsVal) - 1.0)*h;

      DMPlexVertexRestoreCells(lsDM, vert, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

        // Calculate unit normal vector based on the updated level set values at the vertices
        PetscScalar *n;
        xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &n) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexCellGradFromVertex(auxDM, cell, auxVec, lsID, 0, n) >> ablate::utilities::PetscUtilities::checkError;


    }
    MPI_Allreduce(MPI_IN_PLACE, &magDiff, 1, MPIU_REAL, MPIU_MAX, auxCOMM);



    // Get the maximum change across all processors. This also acts as a sync point
    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, auxCOMM);



    PetscPrintf(PETSC_COMM_WORLD, "%d: %+e\t%+e\n", iter, maxDiff, magDiff) >> ablate::utilities::PetscUtilities::checkError;
  }

  SaveCellData("normal.txt", cellNormalField, dim, subDomain);
  SaveVertexData("ls0.txt", lsField, subDomain);


  // Set the vertices too-far away as the largest possible value in the domain with the appropriate sign.
  // This is done after the determination of cut-cells so that all vertices associated with cut-cells have been marked.
  PetscReal gMin[3], gMax[3], maxDist = -1.0;
  DMGetBoundingBox(auxDM, gMin, gMax) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt d = 0; d < dim; ++d) {
    maxDist = PetscMax(maxDist, gMax[d] - gMin[d]);
  }
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);

    // Only worry about cells to far away
    if ( cellMask[c] == 0 ) {
      const PetscScalar *vofVal;
      xDMPlexPointLocalRead(solDM, cell, vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

      PetscReal sgn = PetscSignReal(0.5 - (*vofVal));

      PetscInt nv, *verts;
      DMPlexCellGetVertices(solDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

      for (PetscInt v = 0; v < nv; ++v) {
        PetscInt id = reverseVertRange.GetIndex(verts[v]);
        if (vertMask[id] == 0) {
          PetscScalar *lsVal;
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
        PetscScalar *lsVal;
        xDMPlexPointLocalRef(auxDM, verts[v], lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;
        c0 += *lsVal;
      }
      c0 /= nv;
      DMPlexCellRestoreVertices(solDM, cutCell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;


      PetscInt nCells, *cells;
      DMPlexGetNeighbors(solDM, cutCell, nLevels, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

      for (PetscInt i = 0; i < nCells; ++i) {

        PetscInt cellID = reverseCellRange.GetIndex(cells[i]);
        if (cellMask[cellID]<0) {
          cellMask[cellID] = 1; // Mark as a cell where cell-centered gradients are needed
        }

        PetscInt nv, *verts;
        DMPlexCellGetVertices(solDM, cells[i], &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

        PetscScalar *coords;
        DMPlexVertexGetCoordinates(solDM, nv, verts, &coords) >> ablate::utilities::PetscUtilities::checkError;

        for (PetscInt v = 0; v < nv; ++v) {
          PetscInt id = reverseVertRange.GetIndex(verts[v]);
          if (vertMask[id]<0) {
            vertMask[id] = 1;
          }

          if (vertMask[id]==1) {

            // As an initial guess at the signed-distance function use the distance from the cut-cell center to this vertex
            PetscReal dist = 0.0;
            for (PetscInt d = 0; d < dim; ++d) {
              dist += PetscSqr(x0[d] - coords[v*dim + d]);
            }
            dist = PetscSqrtReal(dist);

            PetscScalar *lsVal;
            xDMPlexPointLocalRef(auxDM, verts[v], lsID, auxArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;

            if (dist < PetscAbs(*lsVal)) {
              PetscScalar *vofVal;
              xDMPlexPointLocalRead(solDM, cells[i], vofID, solArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;
              PetscReal sgn = (*vofVal < 0.5 ? +1.0 : -1.0);
              *lsVal = sgn*dist;
              closestCell[id] = cutCell;
            }
          }
        }

        DMPlexVertexRestoreCoordinates(solDM, nv, verts, &coords) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexCellRestoreVertices(solDM, cells[i], &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
      }
      DMPlexRestoreNeighbors(solDM, cutCell, nLevels, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
    }
  }

  VecRestoreArrayRead(solVec, &solArray) >> ablate::utilities::PetscUtilities::checkError;

  subDomain->UpdateAuxLocalVector();


  const PetscInt vertexNormalID = vertexNormalField->id;
  const PetscInt curvID = curvField->id;

  maxDiff = 1.0;
  iter = 0;
  while (maxDiff>1e-2 && iter<100) {
    ++iter;

    // Determine the current gradient at cells that need updating
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      if (cellMask[c] > 1) {
        PetscInt cell = cellRange.GetPoint(c);
        PetscReal *g;
        xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &g) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexCellGradFromVertex(auxDM, cell, auxVec, lsID, 0, g);
      }
    }
  }

  // Set the vertices too-far away as the largest possible value in the domain with the appropriate sign
  PetscReal gMin[3], gMax[3], maxDist = -1.0;
  DMGetBoundingBox(lsDM, gMin, gMax) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt d = 0; d < dim; ++d) {
    maxDist = PetscMax(maxDist, gMax[d] - gMin[d]);
  }
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);

        xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &phi) >> ablate::utilities::PetscUtilities::checkError;

        DMPlexVertexGradFromVertex(auxDM, vert, auxVec, lsID, 0, g) >> ablate::utilities::PetscUtilities::checkError;

        VertexUpwindGrad(auxDM, auxArray, cellNormalID, vert, PetscSignReal(*phi), g);

        PetscReal nrm = ablate::utilities::MathUtilities::MagVector(dim, g);

        // If the vertex is near the edge of a domain when running in parallel the gradient will be zero, so ignore it.
        if ( nrm > 0.0 ) {

          PetscReal mag = nrm - 1.0;

          PetscReal s = PetscSignReal(*phi);

          tempLS[v] = *phi - h*s*mag;

          maxDiff = PetscMax(maxDiff, PetscAbsReal(mag));
        }
      }
      DMPlexCellRestoreVertices(vofDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
    }
  }

    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
      if (vertMask[v] > 1) {
        PetscInt vert = vertRange.GetPoint(v);
        PetscReal *phi;
        xDMPlexPointLocalRef(auxDM, vert, lsID, auxArray, &phi) >> ablate::utilities::PetscUtilities::checkError;
        *phi = tempLS[v];
      }
    }

     // Get the maximum change across all processors. This also acts as a sync point
    MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPIU_REAL, MPIU_MAX, auxCOMM);

    subDomain->UpdateAuxLocalVector();

    PetscPrintf(PETSC_COMM_WORLD, "%3d: %e\n", iter, maxDiff);
  }



SaveVertexData("ls1.txt", lsField, subDomain);

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
//            const PetscReal *phi;
//            xDMPlexPointLocalRead(auxDM, verts[i], lsID, auxArray, &phi);
//            tempLS[v] += *phi;
//          }
//        }


//        tempLS[v] /= nv;
////        const PetscReal *phi;
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
//        PetscReal *phi;
//        xDMPlexPointLocalRef(auxDM, vert, lsID, auxArray, &phi) >> ablate::utilities::PetscUtilities::checkError;
//        *phi = tempLS[v];
//      }
//    }
//  }


        PetscReal s = PetscSignReal(*phi);

SaveVertexData("ls2.txt", lsField, subDomain);


  // Calculate unit normal vector based on the updated level set values at the vertices
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    if (cellMask[c] > 0) {

        mag = PetscAbsReal(mag);
        diff = PetscMax(diff, mag);

      PetscScalar *n;
      xDMPlexPointLocalRef(auxDM, cell, cellNormalID, auxArray, &n);
      DMPlexCellGradFromVertex(auxDM, cell, auxVec, lsID, 0, n) >> ablate::utilities::PetscUtilities::checkError;
      ablate::utilities::MathUtilities::NormVector(dim, n);
    }
  }

  // Calculate vertex-based unit normal vector based on the updated level set values at the vertices
  for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

    if (vertMask[v] > 0) {
      PetscInt vert = vertRange.GetPoint(v);

      PetscScalar *n;
      xDMPlexPointLocalRef(auxDM, vert, vertexNormalID, auxArray, &n);
      DMPlexVertexGradFromVertex(auxDM, vert, auxVec, lsID, 0, n) >> ablate::utilities::PetscUtilities::checkError;
      ablate::utilities::MathUtilities::NormVector(dim, n);
    }
  }

  // Curvature
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    if (cellMask[c] == 1 ) {
      PetscInt cell = cellRange.GetPoint(c);

      PetscScalar *H;
      xDMPlexPointLocalRef(auxDM, cell, curvID, auxArray, &H);

      *H = ablate::levelSet::geometry::Curvature(rbf, lsField, cell);

//      PetscReal cx = rbf->EvalDer(lsField, cell, 1, 0, 0);
//      PetscReal cy = rbf->EvalDer(lsField, cell, 0, 1, 0);
//      PetscReal cxx = rbf->EvalDer(lsField, cell, 2, 0, 0);
//      PetscReal cyy = rbf->EvalDer(lsField, cell, 0, 2, 0);
//      PetscReal cxy = rbf->EvalDer(lsField, cell, 1, 1, 0);

//      *H = (cxx*cy*cy + cyy*cx*cx - 2.0*cxy*cx*cy)/pow(cx*cx+cy*cy,1.5);

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

  SaveCellData("curv0.txt", curvField, 1, subDomain);

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

      PetscReal *vertexH;
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
            const PetscReal *cellH;
            xDMPlexPointLocalRead(auxDM, cells[c], curvID, auxArray, &cellH);

            *vertexH += *cellH;
          }
        }

        *vertexH /= nc;

        DMPlexVertexRestoreCells(auxDM, v, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;
      }
      else {
        PetscReal *cellH;
        xDMPlexPointLocalRef(auxDM, closestCell[v], curvID, auxArray, &cellH);
        *vertexH = *cellH;
      }
    }
  }
  VecRestoreArray(workVec, &workArray);

  SaveVertexData(auxDM, workVec, "vertexH0.txt", lsField, subDomain);

  maxDiff = 1.0;
  iter = 0;
  while ( maxDiff>1e-2 && iter<10) {
    ++iter;

    VecGetArray(workVec, &workArray);

    // Curvature gradient at the cell-center
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      if (cellMask[c] > 0) {
        PetscInt cell = cellRange.GetPoint(c);
        PetscScalar *g;
        xDMPlexPointLocalRef(auxDM, cell, cellNormalID, workArray, &g) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexCellGradFromVertex(auxDM, cell, workVec, vertexCurvID, 0, g);
      }
    }

    maxDiff = -1.0;


    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
      if (vertMask[v] > 0) {
        PetscInt vert = vertRange.GetPoint(v);
        PetscReal g[dim];
        const PetscReal *phi, *n;
        xDMPlexPointLocalRead(auxDM, vert, lsID, auxArray, &phi) >> ablate::utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRead(auxDM, vert, vertexNormalID, auxArray, &n) >> ablate::utilities::PetscUtilities::checkError;
        for (PetscInt d = 0; d < dim; ++d) g[d] = n[d];

        VertexUpwindGrad(auxDM, workArray, cellNormalID, vert, PetscSignReal(*phi), g);

        PetscReal dH = 0.0;
        for (PetscInt d = 0; d < dim; ++d) dH += g[d]*n[d];


        PetscReal *H;
        xDMPlexPointLocalRef(auxDM, vert, vertexCurvID, workArray, &H);

//        PetscReal Hss = 0.0;
//        for(PetscInt d = 0; d < dim; ++d) {
//          PetscReal g[3];
//          DMPlexVertexGradFromCell(auxDM, vert, workVec, cellNormalID, d, g);
//          Hss += g[d];
//        }
//        *H += 0.1*h*h*Hss;


//          PetscReal s = PetscSignReal(*phi);
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

    PetscPrintf(PETSC_COMM_WORLD, "Extension %3d: %e\n", iter, maxDiff);

  }

  SaveVertexData(auxDM, workVec, "vertexH1.txt", lsField, subDomain);
  DMRestoreLocalVector(auxDM, &workVec);
  DMRestoreGlobalVector(auxDM, &workVecGlobal);


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

  VecRestoreArray(auxVec, &auxArray);
  VecRestoreArray(auxVec, &auxArray);

}
