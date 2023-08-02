#ifndef ABLATELIBRARY_RBF_HPP
#define ABLATELIBRARY_RBF_HPP
#include <petsc.h>
#include <petsc/private/hashmapi.h>
#include "domain/range.hpp"  // For domain::Range
#include "domain/subDomain.hpp"
#include "utilities/petscSupport.hpp"

#define __RBF_DEFAULT_POLYORDER 3

namespace ablate::domain::rbf {

class RBF {
   private:
    std::shared_ptr<ablate::domain::SubDomain> subDomain = nullptr;

    // Radial Basis Function type and parameters
    const int polyOrder = 4;

    PetscInt nPoly = -1;               // The number of polynomial components to include
    PetscInt minNumberCells = -1;      // Minimum number of cells-vertices needed to compute the RBF
    PetscBool useCells = PETSC_FALSE;  // Use vertices or edges/faces when computing neighbor cells/vertices
    const bool useNeighborVertices;    // If it is true formulates the RBF based on vertices surrounding a cell, otherwise will use cells surrounding a cell

    // Information from the subDomain cell range
    PetscInt cStart = 0, cEnd = 0;  // The cell range
    PetscInt *cellList;             // List of the cells to compute over. Length of cEnd - cStart

    // Derivative data
    const bool hasDerivatives;
    PetscInt nDer = 0;                     // Number of derivative stencils which are pre-computed
    PetscInt *dxyz = nullptr;              // The derivatives which have been setup
    PetscHMapI hash = nullptr;             // Hash of the derivative
    PetscInt *nStencil = nullptr;          // Length of each stencil. Needed for both derivatives and interpolation.
    PetscInt **stencilList = nullptr;      // IDs of the points in the stencil. Needed for both derivatives and interpolation.
    PetscReal **stencilWeights = nullptr;  // Weights of the points in the stencil. Needed only for derivatives.
    PetscReal **stencilXLocs = nullptr;    // Locations wrt a cell center. Needed only for interpolation.

    // The derivative->key map for the hash
    PetscInt derivativeKey(PetscInt dx, PetscInt dy, PetscInt dz) const { return (100 * dx + 10 * dy + dz); };

    // Setup the derivative stencil at a point. There is no need for anyone outside of RBF to call this
    void SetupDerivativeStencils(PetscInt c);

    const bool hasInterpolation;
    Mat *RBFMatrix = nullptr;

    // Compute the LU-decomposition of the augmented RBF matrix given a cell list.
    void Matrix(const PetscInt c);

    void CheckField(const ablate::domain::Field *field);  // Checks whether the field is SOL or AUX

   protected:
    PetscReal DistanceSquared(PetscInt dim, PetscReal x[], PetscReal y[]);
    PetscReal DistanceSquared(PetscInt dim, PetscReal x[]);
    void Loc3D(PetscInt dim, PetscReal xIn[], PetscReal x[3]);

   public:
    explicit RBF(int polyOrder = 4, bool hasDerivatives = true, bool hasInterpolation = true, bool useNeighborVertices = false);

    virtual ~RBF();

    /** SubDomain Register and Setup **/
    void Initialize(ablate::domain::Range cellRange);
    void Setup(std::shared_ptr<ablate::domain::SubDomain> subDomain);

    // Derivative stuff
    /**
     * Set the derivatives to use
     * @param numDer - Number of derivatives to set
     * @param dx, dy, dz - Lists of length numDer indicating the derivatives
     * @param useCellsLocal - Use common cells when determining neighbors. If false then use common edges.
     */
    void SetDerivatives(PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[], PetscBool useCellsLocal);

    /**
     * Set the derivatives to use, defaulting to useCells=False
     * @param numDer - Number of derivatives to set
     * @param dx, dy, dz - Lists of length numDer indicating the derivatives
     */
    void SetDerivatives(PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[]);

    /**
     * Setup all derivatives in the subdomain.
     */
    void SetupDerivativeStencils();  // Setup all derivative stencils. Useful if someone wants to remove setup cost when testing

    /**
     * Return the derivative of a field at a given location
     * @param field - The field to take the derivative of
     * @param c - The location in ablate::domain::Range
     * @param dx, dy, dz - The derivative
     */
    PetscReal EvalDer(const ablate::domain::Field *field, PetscInt c, PetscInt dx, PetscInt dy, PetscInt dz);  // Evaluate a derivative

    /**
     * Return the derivative of a field at a given location
     * @param field - The field to take the derivative of
     * @param f - The local vector containing the data
     * @param c - The location in ablate::domain::Range
     * @param dx, dy, dz - The derivative
     */
    PetscReal EvalDer(const ablate::domain::Field *field, Vec f, PetscInt c, PetscInt dx, PetscInt dy, PetscInt dz);  // Evaluate a derivative

    // Interpolation stuff
    /**
     * Return the interpolation of a field at a given location
     * @param field - The field to interpolate
     * @param f - The local vector containing the data
     * @param xEval - The location where to perform the interpolation
     */
    PetscReal Interpolate(const ablate::domain::Field *field, Vec f, PetscReal xEval[3]);

    /**
     * Return the interpolation of a field at a given location
     * @param field - The field to interpolate
     * @param xEval - The location where to perform the interpolation
     */
    PetscReal Interpolate(const ablate::domain::Field *field, PetscReal xEval[3]);

    // These will be overwritten in the derived classes
    /**
     * The RBF kernel value between two points
     * @param dim
     * @param x
     * @param y
     */
    virtual PetscReal RBFVal(PetscInt dim, PetscReal x[], PetscReal y[]) = 0;  // Radial function evaluated using the distance between two points

    /**
     * The RBF kernel derivative between at a location
     * @param dim
     * @param x
     * @param dx, dy, dz - The derivative
     */
    virtual PetscReal RBFDer(PetscInt dim, PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) = 0;  // Derivative of the radial function assuming that the center point is at zero.

    /**
     * The RBF kernel type
     */
    virtual std::string_view type() const = 0;
};

}  // namespace ablate::domain::rbf

#endif  // ABLATELIBRARY_RBF_HPP
