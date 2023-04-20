#ifndef ABLATELIBRARY_LEVELSETUTILITIES_HPP
#define ABLATELIBRARY_LEVELSETUTILITIES_HPP

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
 * @param c0 - The function value at the cell center
 * @param g - The gradient at the cell center
 */
  void CellCenterValGrad(DM dm, const PetscInt p, PetscReal *c, PetscReal *c0, PetscReal *g);





}  // namespace ablate::levelSet::Utilities
#endif  // ABLATELIBRARY_LEVELSETUTILITIES_HPP
