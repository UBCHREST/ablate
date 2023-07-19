#include <petsc.h>
#include <memory>
#include "mathFunctions/functionWrapper.hpp"
#include "utilities/petscUtilities.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/constants.hpp"
#include "levelSetUtilities.hpp"
#include "LS-VOF.hpp"
#include "utilities/petscSupport.hpp"
#include "cellGrad.hpp"
#include "domain/range.hpp"
#include "domain/reverseRange.hpp"

#define xexit(S, ...) {PetscFPrintf(MPI_COMM_WORLD, stderr, \
  "\x1b[1m(%s:%d, %s)\x1b[0m\n  \x1b[1m\x1b[90mexiting:\x1b[0m " S "\n", \
  __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); exit(0);}


void ablate::levelSet::Utilities::CellValGrad(DM dm, const PetscInt p, PetscReal *c, PetscReal *c0, PetscReal *g) {
  DMPolytopeType    ct;
  PetscInt          Nc;
  PetscReal         *coords = NULL;
  const PetscScalar *array;
  PetscBool         isDG;
  PetscReal         x0[3];

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

  PetscInt          nv, *verts;
  const PetscScalar *fvals, *v;
  PetscScalar       *c;

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

  DM  dm = subDomain->GetFieldDM(*field);
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
    * @param dm - Domain of the data
    * @param gradArray - Array storing the cell-center gradients
    * @param cellToIndex - Petsc AO that convertes from DMPlex ordering to the index location in gradArray
    * @param p - Vertex id
    * @param direction - The direction to be considered upwind. +1 for standard upwind, -1 of downwind
    * @param g - On input the surface area normal at a vertex. On output the upwind gradient at p
    */
void ablate::levelSet::Utilities::VertexUpwindGrad(DM dm, PetscReal *gradArray, AO cellToIndex, const PetscInt p, const PetscReal direction, PetscReal *g) {
  // Given an array which stores cell-centered gradients compute the upwind direction
  // The upwind direction is determined using the dot product between the vector u and the vector connecting the cell-center
  //    and the vertex

  PetscInt          dim;
  PetscReal         weightTotal = 0.0;
  PetscScalar       coords[3], n[3];

  DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

  ablate::utilities::MathUtilities::NormVector(dim, g, n);

  DMPlexComputeCellGeometryFVM(dm, p, NULL, coords, NULL) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt d = 0; d < dim; ++d) {
    g[d] = 0.0;
  }


  // Obtain all cells which use this vertex
  PetscInt nCells, *cells;
  DMPlexVertexGetCells(dm, p, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

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

        weightTotal += dot;

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

    // Error checking
  if ( PetscAbs(weightTotal) < ablate::utilities::Constants::small ) {
    throw std::runtime_error("ablate::levelSet::Utilities::VertexUpwindGrad encounted a situation where there are no upwind cells");
  }

  DMPlexVertexRestoreCells(dm, p, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt d = 0; d < dim; ++d) {
    g[d] /= weightTotal;
  }
}

// Given a level set and normal at the cell center compute the level set values at the vertices assuming a straight interface
void ablate::levelSet::Utilities::VertexLevelSet_LS(DM dm, const PetscInt p, const PetscReal c0, const PetscReal *n, PetscReal **c) {
  PetscInt          dim, Nc, nVerts, i, j;
  PetscReal         x0[3] = {0.0, 0.0, 0.0};
  PetscReal         *coords = NULL;
  const PetscScalar *array;
  PetscBool         isDG;

  DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

  // The cell center
  DMPlexComputeCellGeometryFVM(dm, p, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

  // Coordinates of the cell vertices
  DMPlexGetCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;

  // Number of vertices
  nVerts = Nc/dim;

  if (*c == NULL) {
    PetscMalloc1(nVerts, c) >> ablate::utilities::PetscUtilities::checkError;
  }

  // The level set value of each vertex. This assumes that the interface is a line/plane
  //    with the given unit normal.
  for (i = 0; i < nVerts; ++i) {
    (*c)[i] = c0;
    for (j = 0; j < dim; ++j) {
      (*c)[i] += n[j]*(coords[i*dim + j] - x0[j]);
    }
  }

  DMPlexRestoreCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;
}

// Given a cell VOF and normal at the cell center compute the level set values at the vertices assuming a straight interface
void ablate::levelSet::Utilities::VertexLevelSet_VOF(DM dm, const PetscInt p, const PetscReal targetVOF, const PetscReal *n, PetscReal **c) {

  PetscReal vof;  // current VOF of the cell
  PetscReal area; // length (2D) or area (3D) of the cell face
  PetscReal cellVolume; // Area (2D) or volume (3D) of the cell
  const PetscReal tol = 1e-8;
  PetscInt i;
  PetscReal offset;
  PetscReal vofError;
  PetscInt  nv;


  // Get the number of vertices for the cell
  DMPlexCellGetNumVertices(dm, p, &nv) >> ablate::utilities::PetscUtilities::checkError;

  // Get an initial guess at the vertex level set values assuming that the interface passes through the cell-center.
  // Also allocates c if c==NULL on entry
  ablate::levelSet::Utilities::VertexLevelSet_LS(dm, p, 0.0, n, c);

  // Get the resulting VOF from the initial guess
  ablate::levelSet::Utilities::VOF(dm, p, *c, &vof, &area, &cellVolume);
  vofError = targetVOF - vof;

  while ( fabs(vofError) > tol ) {

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


  DMPolytopeType    ct;
  PetscInt          Nc;
  PetscReal         *coords = NULL;
  const PetscScalar *array;
  PetscBool         isDG;

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
void ablate::levelSet::Utilities::VOF(DM dm, PetscInt p, std::shared_ptr<ablate::mathFunctions::MathFunction> phi, PetscReal *vof, PetscReal *area, PetscReal *vol) {

  PetscInt          dim, Nc, nVerts, i;
  PetscReal         *c = NULL, *coords = NULL;
  const PetscScalar *array;
  PetscBool         isDG;

  DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

  // Coordinates of the cell vertices
  DMPlexGetCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;

  // Number of vertices
  nVerts = Nc/dim;

  PetscMalloc1(nVerts, &c) >> ablate::utilities::PetscUtilities::checkError;

  // The level set value of each vertex. This assumes that the interface is a line/plane
  //    with the given unit normal.
  for (i = 0; i < nVerts; ++i) {
   c[i] = phi->Eval(&coords[i*dim], dim, 0.0);
  }

  DMPlexRestoreCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::utilities::PetscUtilities::checkError;

  ablate::levelSet::Utilities::VOF(dm, p, c, vof, area, vol);  // Do the actual calculation.

  PetscFree(c) >> ablate::utilities::PetscUtilities::checkError;
}


// Return the VOF in a cell where the level set is defined at vertices
void ablate::levelSet::Utilities::VOF(std::shared_ptr<ablate::domain::SubDomain> subDomain, PetscInt cell, const ablate::domain::Field *lsField,
      PetscReal *vof, PetscReal *area, PetscReal *vol) {

    DM   dm = subDomain->GetFieldDM(*lsField);
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




void SaveVertexData(const char fname[255], const ablate::domain::Field *field, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

  ablate::domain::Range range;
  PetscReal    *array, *val;
  Vec           vec = subDomain->GetVec(*field);
  DM            dm  = subDomain->GetFieldDM(*field);
  PetscInt      dim = subDomain->GetDimensions();

  ablate::domain::GetRange(dm, nullptr, 0, range);

  VecGetArray(vec, &array) >> ablate::utilities::PetscUtilities::checkError;

  FILE *f1 = fopen(fname, "w");

  for (PetscInt v = range.start; v < range.end; ++v) {
    PetscInt vert = range.points ? range.points[v] : v;
    PetscScalar *coords;

    DMPlexPointLocalFieldRef(dm, vert, field->id, array, &val) >> ablate::utilities::PetscUtilities::checkError;

    DMPlexVertexGetCoordinates(dm, 1, &vert, &coords);

    for (PetscInt d = 0; d < dim; ++d) {
      fprintf(f1, "%+.16e\t", coords[d]);
    }
    fprintf(f1, "%+.16ef\n", *val);

    DMPlexVertexRestoreCoordinates(dm, 1, &vert, &coords);
  }

  fclose(f1);

  VecRestoreArray(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
  ablate::domain::RestoreRange(range);
}






// Compute the level-set field that corresponds to a given VOF field
// The steps are:
//  1 - Determine the level-set field in cells containing a VOF value between 0 and 1
//  2 - Mark the required number of vertices (based on the cells) next to the interface cells
//  3 - Iterate over vertices EXCEPT for those with cut-cells until converged
//  4 - We may want to look at a fourth step which improve the accuracy
void ablate::levelSet::Utilities::Reinitialize(std::shared_ptr<ablate::domain::SubDomain> subDomain, const ablate::domain::Field *vofField, const PetscInt nLevels, const ablate::domain::Field *lsField) {

  // Note: Need to write a unit test where the vof and ls fields aren't in the same DM, e.g. one is a SOL vector and one is an AUX vector.

  DM vofDM = subDomain->GetFieldDM(*vofField);
  DM lsDM = subDomain->GetFieldDM(*lsField);
  const PetscInt lsID = lsField->id;
  const PetscInt vofID = vofField->id;
  Vec vofVec = subDomain->GetVec(*vofField);
  Vec lsVec = subDomain->GetVec(*lsField);
  PetscScalar *lsArray;
  const PetscScalar *vofArray;
  const PetscInt dim = subDomain->GetDimensions();   // VOF and LS subdomains must have the same dimension. Can't think of a reason they wouldn't.


  ablate::domain::Range cellRange, vertRange;
  subDomain->GetCellRange(nullptr, cellRange);
  subDomain->GetRange(nullptr, 0, vertRange);


  // Get the point->index mapping for cells
  ablate::domain::ReverseRange reverseVertRange = ablate::domain::ReverseRange(vertRange);
  ablate::domain::ReverseRange reverseCellRange = ablate::domain::ReverseRange(cellRange);


  PetscInt *vertMask, *cellMask;
  DMGetWorkArray(lsDM, vertRange.end - vertRange.start, MPIU_INT, &vertMask) >> ablate::utilities::PetscUtilities::checkError;
  DMGetWorkArray(vofDM, cellRange.end - cellRange.start, MPIU_INT, &cellMask) >> ablate::utilities::PetscUtilities::checkError;
  vertMask -= vertRange.start; // offset so that we can use start->end
  cellMask -= cellRange.start; // offset so that we can use start->end
  for (PetscInt i = vertRange.start; i < vertRange.end; ++i) {
    vertMask[i] = -1; //  Ignore the vertex
  }
  for (PetscInt i = cellRange.start; i < cellRange.end; ++i) {
    cellMask[i] = -1; // Ignore the cell
  }



  VecGetArray(lsVec, &lsArray) >> ablate::utilities::PetscUtilities::checkError;
  for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {
    PetscInt vert = vertRange.GetPoint(v);
    PetscScalar *val;
    xDMPlexPointLocalRef(lsDM, vert, lsID, lsArray, &val) >> ablate::utilities::PetscUtilities::checkError;
    *val = PETSC_MAX_REAL;
  }

  PetscReal h;
  DMPlexGetMinRadius(vofDM, &h) >> ablate::utilities::PetscUtilities::checkError;

  VecGetArrayRead(vofVec, &vofArray) >> ablate::utilities::PetscUtilities::checkError;
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.GetPoint(c);

    PetscScalar *vofVal;
    xDMPlexPointLocalRead(vofDM, cell, vofID, vofArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

    // Only worry about cut-cells
    if ( ((*vofVal) > ablate::utilities::Constants::small) && ((*vofVal) < (1.0 - ablate::utilities::Constants::small)) ) {

      cellMask[c] = 0;  // Mark as a cut-cell

      PetscInt nv, *verts;
      DMPlexCellGetVertices(vofDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

      for (PetscInt v = 0; v < nv; ++v) {
        // Mark as a cut-cell vertex
        vertMask[reverseVertRange.GetIndex(verts[v])] = 0; // Mark vertex as associated with a cut-cell
      }

      DMPlexCellRestoreVertices(vofDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
    }
  }

  for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

    if (vertMask[v]==0) {

      PetscInt vert = vertRange.GetPoint(v);

      PetscReal x0[3];
      DMPlexComputeCellGeometryFVM(lsDM, vert, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

      PetscScalar *lsVal;
      xDMPlexPointLocalRef(lsDM, vert, lsID, lsArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;



      PetscInt nCells, *cells;
      DMPlexVertexGetCells(lsDM, vert, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

      *lsVal = 0.0;
      PetscReal totalWeight = 0.0;
      for (PetscInt i = 0; i < nCells; ++i) {
        PetscScalar *vofVal, x[3];
        xDMPlexPointLocalRead(vofDM, cells[i], vofID, vofArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexComputeCellGeometryFVM(vofDM, cells[i], NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;

        ablate::utilities::MathUtilities::Subtract(dim, x, x0, x);
        PetscReal wt = ablate::utilities::MathUtilities::MagVector(dim, x);

        *lsVal += (*vofVal)*wt;
        totalWeight += wt;
      }
      *lsVal /= totalWeight;

      *lsVal = -2.0*(2.0*(*lsVal) - 1.0)*h;

      DMPlexVertexRestoreCells(lsDM, vert, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

//*lsVal = PetscSqrtReal(PetscSqr(x0[0]) + PetscSqr(x0[1])) - 1.0;


    }
  }





  // Comment out the rest of the code so that we can focus on the cut-cells only

  // Now mark all of the necessary neighboring vertices. Note that this can't be put into the previous loop as all of the vertices
  //    for the cut-cells won't be known yet.
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cutCell = cellRange.GetPoint(c);

    // Only worry about cut-cells
    if ( cellMask[c]==0 ) {
      // Center of the cell
      PetscReal x0[3];
      DMPlexComputeCellGeometryFVM(vofDM, cutCell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

      // Once the neightbor function for vertices is merged this will need to be moved over
      PetscInt nCells, *cells;
      DMPlexGetNeighbors(vofDM, cutCell, nLevels, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cells) >> ablate::utilities::PetscUtilities::checkError;

      for (PetscInt i = 0; i < nCells; ++i) {

        PetscInt cellID = reverseCellRange.GetIndex(cells[i]);
        if (cellMask[cellID]<0) {
          cellMask[cellID] = 1; // Mark as a cell where cell-centered gradients are needed
        }

        PetscInt nv, *verts;
        DMPlexCellGetVertices(vofDM, cells[i], &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

        PetscScalar *coords;
        DMPlexVertexGetCoordinates(vofDM, nv, verts, &coords) >> ablate::utilities::PetscUtilities::checkError;

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
            xDMPlexPointLocalRef(lsDM, verts[v], lsID, lsArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;

            if (dist < PetscAbs(*lsVal)) {
              PetscScalar *vofVal;
              xDMPlexPointLocalRead(vofDM, cells[i], vofID, vofArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;
              PetscReal sgn = (*vofVal < 0.5 ? +1.0 : -1.0);
              *lsVal = sgn*dist;
            }
          }
        }

        DMPlexVertexRestoreCoordinates(vofDM, nv, verts, &coords) >> ablate::utilities::PetscUtilities::checkError;
        DMPlexCellRestoreVertices(vofDM, cells[i], &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
      }

      PetscFree(cells) >> ablate::utilities::PetscUtilities::checkError;

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

    // Only worry about cells to far away
    if ( cellMask[c]==-1 ) {
      PetscScalar *vofVal;
      xDMPlexPointLocalRead(vofDM, cell, vofID, vofArray, &vofVal) >> ablate::utilities::PetscUtilities::checkError;

      PetscReal sgn = PetscSignReal(0.5 - (*vofVal));

      PetscInt nv, *verts;
      DMPlexCellGetVertices(vofDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;

      for (PetscInt v = 0; v < nv; ++v) {
        PetscInt id = reverseVertRange.GetIndex(verts[v]);
        if (vertMask[id]<0) {
          PetscScalar *lsVal;
          xDMPlexPointLocalRef(lsDM, verts[v], lsID, lsArray, &lsVal) >> ablate::utilities::PetscUtilities::checkError;
          *lsVal = sgn*maxDist;
        }
      }
      DMPlexCellRestoreVertices(vofDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
    }
  }

  VecRestoreArrayRead(vofVec, &vofArray) >> ablate::utilities::PetscUtilities::checkError;

  // Create the mapping between DMPlex cell numbering and location in the array storing cell-centered gradients
  PetscInt numCells = 0;
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    numCells += (cellMask[c]>-1);
  }

  PetscInt *cellArray, *indexArray;
  DMGetWorkArray(vofDM, numCells, MPIU_INT, &cellArray) >> ablate::utilities::PetscUtilities::checkError;
  DMGetWorkArray(vofDM, numCells, MPIU_INT, &indexArray) >> ablate::utilities::PetscUtilities::checkError;


  PetscInt i = 0;
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    if (cellMask[c]>-1) {
      cellArray[i] = cellRange.GetPoint(c);
      indexArray[i] = i;
      ++i;

    }
  }

  cellMask += cellRange.start;  // Reset the offset, otherwise DMRestoreWorkArray will return unexpected results
  DMRestoreWorkArray(vofDM, cellRange.end - cellRange.start, MPIU_INT, &cellMask) >> ablate::utilities::PetscUtilities::checkError;
  subDomain->RestoreRange(cellRange);

  AO cellToIndex;
  AOCreateMapping(PETSC_COMM_SELF, numCells, cellArray, indexArray, &cellToIndex) >> ablate::utilities::PetscUtilities::checkError;

  PetscScalar *cellGradArray;
  DMGetWorkArray(vofDM, dim*numCells, MPIU_SCALAR, &cellGradArray) >> ablate::utilities::PetscUtilities::checkError;

  PetscReal diff = 1.0;
  PetscInt it = 0;
  while (diff>1e-2 && it<77e10) {
    ++it;
    for (PetscInt i = 0; i < numCells; ++i) {
      PetscInt cell = cellArray[i];

      ablate::levelSet::Utilities::CellValGrad(lsDM, lsID, cell, lsVec, nullptr, &(cellGradArray[i*dim]));

    }

    diff = 0.0;

    for (PetscInt v = vertRange.start; v < vertRange.end; ++v) {

      if (vertMask[v]==1) {
        PetscInt vert = vertRange.GetPoint(v);
        PetscReal g[dim];
        PetscReal *phi;

        xDMPlexPointLocalRef(lsDM, vert, lsID, lsArray, &phi) >> ablate::utilities::PetscUtilities::checkError;

        DMPlexVertexGradFromVertex(lsDM, vert, lsVec, lsID, 0, g) >> ablate::utilities::PetscUtilities::checkError;

        VertexUpwindGrad(lsDM, cellGradArray, cellToIndex, vert, PetscSignReal(*phi), g);

        PetscReal mag = ablate::utilities::MathUtilities::MagVector(dim, g) - 1.0;

        PetscReal s = PetscSignReal(*phi);

        *phi -= h*s*mag;

        mag = PetscAbsReal(mag);
        diff = PetscMax(diff, mag);

      }
    }
    printf("%e\n", diff);
  }


  DMRestoreWorkArray(vofDM, dim*numCells, MPIU_SCALAR, &cellGradArray) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreWorkArray(vofDM, numCells, MPIU_INT, &cellArray) >> ablate::utilities::PetscUtilities::checkError;
  DMRestoreWorkArray(vofDM, numCells, MPIU_INT, &indexArray) >> ablate::utilities::PetscUtilities::checkError;
  AODestroy(&cellToIndex) >> ablate::utilities::PetscUtilities::checkError;


  vertMask += vertRange.start; // Reset the offset, otherwise DMRestoreWorkArray will return unexpected results
  DMRestoreWorkArray(lsDM, vertRange.end - vertRange.start, MPIU_INT, &vertMask) >> ablate::utilities::PetscUtilities::checkError;
  subDomain->RestoreRange(vertRange);


}
