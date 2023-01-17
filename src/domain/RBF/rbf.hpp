#ifndef ABLATELIBRARY_RBF_HPP
#define ABLATELIBRARY_RBF_HPP
#include <petsc.h>
#include "domain/subDomain.hpp"
#include "solver/solver.hpp" // For solver::Range
#include "rbfSupport.hpp"

#define __RBF_DEFAULT_POLYORDER 4

namespace ablate::domain::rbf {

class RBF {

  private:


    std::shared_ptr<ablate::domain::SubDomain> subDomain;

    // Radial Basis Function type and parameters
    const PetscInt polyOrder = 4;

    PetscInt  nPoly = -1;               // The number of polynomial components to include
    PetscInt  minNumberCells = -1;      // Minimum number of cells needed to compute the RBF
    PetscBool useVertices = PETSC_TRUE; // Use vertices or edges/faces when computing neighbor cells


    // Information from the subDomain cell range
    PetscInt cStart = 0, cEnd = 0;  // The cell range
    PetscInt *cellList;             // List of the cells to compute over. Length of cEnd - cStart


    // Derivative data
    const bool hasDerivatives;
    PetscInt nDer = 0;                      // Number of derivative stencils which are pre-computed
    PetscInt *dxyz = nullptr;               // The derivatives which have been setup
    PetscInt *nStencil = nullptr;           // Length of each stencil. Needed for both derivatives and interpolation.
    PetscInt **stencilList = nullptr;       // IDs of the points in the stencil. Needed for both derivatives and interpolation.
    PetscReal **stencilWeights = nullptr;   // Weights of the points in the stencil. Needed only for derivatives.
    PetscReal **stencilXLocs = nullptr;     // Locations wrt a cell center. Needed only for interpolation.

    // Setup the derivative stencil at a point. There is no need for anyone outside of RBF to call this
    void SetupDerivativeStencils(PetscInt c);

    const bool hasInterpolation;
    Mat *RBFMatrix = nullptr;

    // Compute the LU-decomposition of the augmented RBF matrix given a cell list.
    void Matrix(const PetscInt c, PetscReal **x, Mat *LUA);


  protected:
    PetscReal DistanceSquared(PetscReal x[], PetscReal y[]);
    PetscReal DistanceSquared(PetscReal x[]);

  public:

    RBF(PetscInt polyOrder, bool hasDerivatives, bool hasInterpolation);

    ~RBF();

    /** SubDomain Register and Setup **/
    void Initialize(solver::Range cellRange);
    void Setup(std::shared_ptr<ablate::domain::SubDomain> subDomain);

    // Derivative stuff
    void SetDerivatives(PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[], PetscBool useVertices);
    void SetDerivatives(PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[]);
    void SetupDerivativeStencils();   // Setup all derivative stencils. Useful if someone wants to remove setup cost when testing

    PetscReal EvalDer(const ablate::domain::Field *field, PetscInt c, PetscInt dx, PetscInt dy, PetscInt dz);  // Evaluate a derivative
    PetscReal EvalDer(DM dm, Vec vec, PetscInt c, PetscInt dx, PetscInt dy, PetscInt dz);

    // Interpolation stuff
    PetscReal Interpolate(const ablate::domain::Field *field, PetscReal xEval[3]);
    PetscReal Interpolate(DM dm, Vec f, PetscReal xEval[3]);

    // To make setup easier
    void GetRange(std::shared_ptr<ablate::domain::SubDomain> subDomain, const std::shared_ptr<ablate::domain::Region> region, PetscInt depth, ablate::solver::Range &range) const;
    void GetCellRange(std::shared_ptr<ablate::domain::SubDomain> subDomain, const std::shared_ptr<ablate::domain::Region> region, ablate::solver::Range &range) const;
    void RestoreRange(ablate::solver::Range &range) const;

    // These will be overwritten in the derived classes
    virtual PetscReal RBFVal(PetscReal x[], PetscReal y[]) = 0;   // Radial function evaluated using the distance between two points
    virtual PetscReal RBFDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) = 0; // Derivative of the radial function assuming that the center point is at zero.


};

}  // namespace ablate::domain

#endif  // ABLATELIBRARY_RBF_HPP
