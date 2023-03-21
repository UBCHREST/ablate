#ifndef ABLATELIBRARY_RBF_HPP
#define ABLATELIBRARY_RBF_HPP
#include <petsc.h>
#include "domain/subDomain.hpp"
#include "rbfSupport.hpp"
#include "domain/range.hpp"  // For domain::Range

#define __RBF_DEFAULT_POLYORDER 3

namespace ablate::domain::rbf {

class RBF {
   private:
    std::shared_ptr<ablate::domain::SubDomain> subDomain = nullptr;

    // Radial Basis Function type and parameters
    const int polyOrder = 4;

    PetscInt nPoly = -1;                 // The number of polynomial components to include
    PetscInt minNumberCells = -1;        // Minimum number of cells needed to compute the RBF
    PetscBool useVertices = PETSC_TRUE;  // Use vertices or edges/faces when computing neighbor cells

    // Information from the subDomain cell range
    PetscInt cStart = 0, cEnd = 0;  // The cell range
    PetscInt *cellList;             // List of the cells to compute over. Length of cEnd - cStart

    // Derivative data
    const bool hasDerivatives;
    PetscInt nDer = 0;                     // Number of derivative stencils which are pre-computed
    PetscInt *dxyz = nullptr;              // The derivatives which have been setup
    PetscInt *nStencil = nullptr;          // Length of each stencil. Needed for both derivatives and interpolation.
    PetscInt **stencilList = nullptr;      // IDs of the points in the stencil. Needed for both derivatives and interpolation.
    PetscReal **stencilWeights = nullptr;  // Weights of the points in the stencil. Needed only for derivatives.
    PetscReal **stencilXLocs = nullptr;    // Locations wrt a cell center. Needed only for interpolation.

    // Setup the derivative stencil at a point. There is no need for anyone outside of RBF to call this
    void SetupDerivativeStencils(PetscInt c);

    const bool hasInterpolation;
    Mat *RBFMatrix = nullptr;

    // Compute the LU-decomposition of the augmented RBF matrix given a cell list.
    void Matrix(const PetscInt c);

   protected:
    PetscReal DistanceSquared(PetscInt dim, PetscReal x[], PetscReal y[]);
    PetscReal DistanceSquared(PetscInt dim, PetscReal x[]);
    void Loc3D(PetscInt dim, PetscReal xIn[], PetscReal x[3]);

   public:
    RBF(int polyOrder = 4, bool hasDerivatives = true, bool hasInterpolation = true);

    virtual ~RBF();

    /** SubDomain Register and Setup **/
    void Initialize(ablate::domain::Range cellRange);
    void Setup(std::shared_ptr<ablate::domain::SubDomain> subDomain);

    // Derivative stuff
    void SetDerivatives(PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[], PetscBool useVertices);
    void SetDerivatives(PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[]);
    void SetupDerivativeStencils();  // Setup all derivative stencils. Useful if someone wants to remove setup cost when testing

    PetscReal EvalDer(const ablate::domain::Field *field, PetscInt c, PetscInt dx, PetscInt dy, PetscInt dz);  // Evaluate a derivative

    // Interpolation stuff
    PetscReal Interpolate(const ablate::domain::Field *field, PetscReal xEval[3]);

    // These will be overwritten in the derived classes
    virtual PetscReal RBFVal(PetscInt dim, PetscReal x[], PetscReal y[]) = 0;                          // Radial function evaluated using the distance between two points
    virtual PetscReal RBFDer(PetscInt dim, PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) = 0;  // Derivative of the radial function assuming that the center point is at zero.
    virtual std::string_view type() const = 0;
};

}  // namespace ablate::domain::rbf

#endif  // ABLATELIBRARY_RBF_HPP
