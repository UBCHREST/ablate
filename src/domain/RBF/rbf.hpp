#ifndef ABLATELIBRARY_RBF_HPP
#define ABLATELIBRARY_RBF_HPP
#include <petsc.h>
#include <string>
#include <vector>
#include <domain/region.hpp>
#include "domain/domain.hpp"
#include "domain/subDomain.hpp"
#include "rbfSupport.hpp"
#include "domain/modifiers/modifier.hpp"



#define __RBF_DEFAULT_POLYORDER 4
#define __RBF_DEFAULT_PARAM 0.1

namespace ablate::domain::rbf {

class RBF {

  private:


    const std::shared_ptr<ablate::domain::SubDomain> subDomain;


    // Radial Basis Function type and parameters
    const PetscInt polyOrder = 4;

    PetscInt nCells = -1;               // Number of cells in ablate::solver::Range

    PetscInt  nPoly = -1;               // The number of polynomial components to include
    PetscInt  minNumberCells = -1;      // Minimum number of cells needed to compute the RBF
    PetscBool useVertices = PETSC_TRUE; // Use vertices or edges/faces when computing neighbor cells


    // Derivative data
    PetscBool hasDerivativeInformation = PETSC_FALSE;
    PetscInt nDer = 0;                      // Number of derivative stencils which are pre-computed
    PetscInt *dxyz = nullptr;               // The derivatives which have been setup
    PetscInt *nStencil = nullptr;           // Length of each stencil. Needed for both derivatives and interpolation.
    PetscInt **stencilList = nullptr;       // IDs of the points in the stencil. Needed for both derivatives and interpolation.
    PetscReal **stencilWeights = nullptr;   // Weights of the points in the stencil. Needed only for derivatives.
    PetscReal **stencilXLocs = nullptr;     // Locations wrt a cell center. Needed only for interpolation.

    // Setup the derivative stencil at a point. There is no need for anyone outside of RBF to call this
    void SetupDerivativeStencils(PetscInt c);

    PetscBool hasInterpolation = PETSC_FALSE;
    Mat *RBFMatrix = nullptr;

    // Compute the LU-decomposition of the augmented RBF matrix given a cell list.
    void Matrix(const PetscInt c, PetscReal **x, Mat *LUA);


  protected:
    PetscReal DistanceSquared(PetscReal x[], PetscReal y[]);
    PetscReal DistanceSquared(PetscReal x[]);

    // These will be overwritten in the derived classes
    virtual PetscReal RBFVal(PetscReal x[], PetscReal y[]) = 0;
    virtual PetscReal RBFDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) = 0;


  public:



    RBF(std::shared_ptr<ablate::domain::SubDomain> subDomain, PetscInt polyOrder);

//    RBF(std::shared_ptr<ablate::domain::SubDomain> subDomain);

    ~RBF();


    /** SubDomain Register and Setup **/
    void Initialize();
    void Setup();
    void Register();
//    void Modify(DM&) override;


    // Derivative stuff
    void SetDerivatives(PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[], PetscBool useVertices);
    void SetDerivatives(PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[]);
    PetscReal EvalDer(const ablate::domain::Field *field, PetscInt c, PetscInt dx, PetscInt dy, PetscInt dz);  // Evaluate a derivative
    void SetupDerivativeStencils();   // Setup all derivative stencils. Useful if someone wants to remove setup cost when testing

    // Interpolation stuff
    void SetInterpolation(PetscBool hasInterpolation);
    PetscReal Interpolate(const ablate::domain::Field *field, PetscInt c, PetscReal xEval[3]);



};

}  // namespace ablate::domain

#endif  // ABLATELIBRARY_RBF_HPP
