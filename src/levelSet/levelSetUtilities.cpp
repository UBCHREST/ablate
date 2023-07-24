#include <petsc.h>
#include <memory>
#include "mathFunctions/functionWrapper.hpp"
#include "utilities/petscUtilities.hpp"
#include "levelSetUtilities.hpp"
#include "LS-VOF.hpp"

// Given a level set and normal at the cell center compute the level set values at the vertices assuming a straight interface
void ablate::levelSet::Utilities::VertexLevelSet(DM dm, const PetscInt p, const PetscReal c0, const PetscReal *nIn, PetscReal **c) {
  PetscInt          dim, Nc, nVerts, i, j;
  PetscReal         x0[3] = {0.0, 0.0, 0.0};
  PetscReal         *coords = NULL;
  PetscScalar       n[3] = {0.0, 0.0, 0.0}, g;
  const PetscScalar *array;
  PetscBool         isDG;

  DMGetDimension(dm, &dim) >> ablate::utilities::PetscUtilities::checkError;

  // The cell center
  DMPlexComputeCellGeometryFVM(dm, p, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

  g = 0.0;
  for (i = 0; i < dim; ++i) g += PetscSqr(nIn[i]);
  g = PetscSqrtReal(g);
  for (i = 0; i < dim; ++i) n[i] = nIn[i]/g;

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
  ablate::levelSet::Utilities::VertexLevelSet(dm, p, c0, nIn, &c);

  ablate::levelSet::Utilities::VOF(dm, p, c, vof, area, vol);  // Do the actual calculation.

  PetscFree(c) >> ablate::utilities::PetscUtilities::checkError;

}

// Returns the VOF for a given cell using an analytic level set equation
// Refer to "Quadrature rules for triangular and tetrahedral elements with generalized functions"
void ablate::levelSet::Utilities::VOF(DM dm, PetscInt p, const std::shared_ptr<ablate::mathFunctions::MathFunction>& phi, PetscReal *vof, PetscReal *area, PetscReal *vol) {

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


//void ablate::levelSet::Utilities::VOFField(DM dm, const ablate::domain::Field *field, std::shared_ptr<ablate::mathFunctions::MathFunction> phi) {



//}
