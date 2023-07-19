#ifndef ABLATELIBRARY_LEVELSETUTILITIES_HPP
#define ABLATELIBRARY_LEVELSETUTILITIES_HPP

#include "domain/subDomain.hpp"
#include "domain/RBF/rbf.hpp"

namespace ablate::levelSet::Utilities {

  /**
   * Calculate the VOF for a cell given a level set and normal at the cell center
   * @param dm - The mesh
   * @param p - Cell id
   * @param c0 - Level set at the cell center
   * @param n - Outward facing normal at the cell center
   * @param vof - The volume-of-fluid
   * @param area - The face length(2D) or area(3D) in the cell
   * @param vol - The area/volume of the entire cell
   */
  void VOF(DM dm, const PetscInt p, const PetscReal c0, const PetscReal *n, PetscReal *vof, PetscReal *area, PetscReal *vol);

  /**
   * Calculate the VOF for a cell given level set values at the vertices
   * @param dm - The mesh
   * @param p - Cell id
   * @param c - Level set values at the cell vertices. Order of values must match that returned by DMPlexGetCellCoordinates.
   * @param vof - The volume-of-fluid
   * @param area - The face length(2D) or area(3D) in the cell
   * @param vol - The area/volume of the entire cell
   */
  void VOF(DM dm, const PetscInt p, PetscReal *c, PetscReal *vof, PetscReal *area, PetscReal *vol);

  /**
   * Calculate the VOF for a cell given an analytic level set function
   * @param dm - The mesh
   * @param p - Cell id
   * @param phi - Function used to calculate the level set values at the vertices
   * @param vof - The volume-of-fluid
   * @param area - The face length(2D) or area(3D) in the cell
   * @param vol - The area/volume of the entire cell
   */
  void VOF(DM dm, PetscInt p, std::shared_ptr<ablate::mathFunctions::MathFunction> phi, PetscReal *vof, PetscReal *area, PetscReal *vol);

  /**
   * Calculate the VOF for a cell given an analytic level set function
   * @param subDomain - Domain of the vertex-based level-set field
   * @param cell - Cell id
   * @param lsField - Field of the level set
   * @param vof - The volume-of-fluid
   * @param area - The face length(2D) or area(3D) in the cell
   * @param vol - The area/volume of the entire cell
   */
  void VOF(std::shared_ptr<ablate::domain::SubDomain> subDomain, PetscInt cell, const ablate::domain::Field *lsField, PetscReal *vof, PetscReal *area, PetscReal *vol);

  /**
   * Return the vertex level set values at a cell's vertices assuming a straight interface in the cell with a given normal vector using a cell-center level set value
   * @param dm - The mesh
   * @param p - Cell id
   * @param c0 - Level set value at the cell center
   * @param nIn - The normal at the cell center
   * @param c - Level set values at the vertex. If the array is NULL on input memory is allocated.
   */
  void VertexLevelSet_LS(DM dm, const PetscInt p, const PetscReal c0, const PetscReal *nIn, PetscReal **c);

  /**
   * Return the vertex level set values at a cell's vertices assuming a straight interface in the cell with a given normal vector using a cell VOF
   * @param dm - The mesh
   * @param p - Cell id
   * @param vof - Level set value at the cell center
   * @param nIn - The normal at the cell center
   * @param c - Level set values at the vertex. If the array is NULL on input memory is allocated.
   */
  void VertexLevelSet_VOF(DM dm, const PetscInt p, const PetscReal vof, const PetscReal *nIn, PetscReal **c);


  /**
    * Cell-wise function value and gradient at a given location
    * @param dm - The mesh
    * @param p - Cell id
    * @param c - Function to find gradient of at cell center. Order of values must match that returned by DMPlexGetCellCoordinates.
    * @param c0 - The function value at x0
    * @param g - The gradient at x0
    */
  void CellValGrad(DM dm, const PetscInt p, PetscReal *c, PetscReal *c0, PetscReal *g);

  /**
    * Cell-wise function value and gradient at a given location
    * @param dm - The mesh
    * @param fid - Field ID of the data
    * @param p - Cell id
    * @param f - The vector containing the data
    * @param c0 - The function value at x0
    * @param g - The gradient at x0
    */
  void CellValGrad(DM dm, const PetscInt fid, const PetscInt p, Vec f, PetscReal *c0, PetscReal *g);

  /**
    * Cell-wise function value and gradient at a given location
    * @param subDomain - Domain of the data
    * @param field - Field containing the vertex data
    * @param p - Cell id
    * @param c0 - The function value at x0
    * @param g - The gradient at x0
    */
  void CellValGrad(std::shared_ptr<ablate::domain::SubDomain> subDomain, const ablate::domain::Field *field, const PetscInt p, PetscReal *c0, PetscReal *g);


  /**
    * Vertex gradient
    * @param subDomain - Domain of the data
    * @param field - Field containing the vertex data
    * @param p - Vertex id
    * @param g - The gradient at p
    */
  void VertexToVertexGrad(std::shared_ptr<ablate::domain::SubDomain> subDomain, const ablate::domain::Field *field, const PetscInt p, PetscReal *g);

//  /**
//    * Vertex gradient using a given cell-centered gradient vector
//    * @param dm - Domain of the data
//    * @param vec - Vector containing the cell-centered gradient information
//    * @param fid - The field ID of the cell-centered gradient information
//    * @param p - Vertex id
//    * @param g - The gradient at p
//    */
//  void VertexGrad(DM dm, Vec vec, const PetscInt fid, const PetscInt p, PetscReal *g);

//  /**
//    * Compute the upwind derivative
//    * @param dm - Domain of the data
//    * @param vec - Vector containing the cell-centered gradient information
//    * @param fid - The field ID of the cell-centered gradient information
//    * @param p - Vertex id
//    * @param direction - The direction to be considered upwind. +1 for standard upwind, -1 of downwind
//    * @param g - The gradient at p
//    */
//  void VertexUpwindGrad(DM dm, Vec vec, const PetscInt fid, const PetscInt p, const PetscReal direction, PetscReal *g);

  /**
    * Compute the upwind derivative
    * @param dm - Domain of the data
    * @param gradArray - Array storing the cell-center gradients
    * @param cellToIndex - Petsc AO that convertes from DMPlex ordering to the index location in gradArray
    * @param p - Vertex id
    * @param direction - The direction to be considered upwind. +1 for standard upwind, -1 of downwind
    * @param g - The gradient at p
    */
  void VertexUpwindGrad(DM dm, PetscReal *gradArray, AO cellToIndex, const PetscInt p, const PetscReal direction, PetscReal *g);

  /**
    * Compute the leve-set field that corresponds to a volume-of-fluid field
    * @param rbf - The cell-centered RBF used to estimate the unit normal
    * @param vofSubDomain - The domain containing the VOF data
    * @param vofField - Location of cell-centered VOF data
    * @param nLevels - The number of layers/levels to use surrounding each interface cell
    * @param lsSubDomain - The domain containing the LS data
    * @param lsField - Location of vertex-based LS data
    */
  void Reinitialize(std::shared_ptr<ablate::domain::SubDomain> subDomain, const ablate::domain::Field *vofField, const PetscInt nLevels, const ablate::domain::Field *lsField);



}  // namespace ablate::levelSet::Utilities
#endif  // ABLATELIBRARY_LEVELSETUTILITIES_HPP
