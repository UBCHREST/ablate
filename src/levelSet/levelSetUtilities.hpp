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
 * @param c - Level set values at the cell vertices
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
void VOF(DM dm, PetscInt p, const std::shared_ptr<ablate::mathFunctions::MathFunction> &phi, PetscReal *vof, PetscReal *area, PetscReal *vol);

/**
 * Return the vertex level set values assuming a straight interface in the cell with a given normal vector.
 * @param dm - The mesh
 * @param p - Cell id
 * @param c0 - Level set value at the cell center
 * @param nIn - The normal at the cell center
 * @param c - Level set values at the vertex. If NULL on input memory is allocated.
 */
void VertexLevelSet(DM dm, const PetscInt p, const PetscReal c0, const PetscReal *nIn, PetscReal **c);

}  // namespace ablate::levelSet::Utilities
#endif  // ABLATELIBRARY_LEVELSETUTILITIES_HPP
