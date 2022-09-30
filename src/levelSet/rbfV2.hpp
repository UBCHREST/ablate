#ifndef ABLATELIBRARY_RBFV2_HPP
#define ABLATELIBRARY_RBFV2_HPP
#include <petsc.h>
#include <string>
#include <vector>
#include "domain/domain.hpp"
#include "solver/solver.hpp"
#include "solver/timeStepper.hpp"
#include "lsSupport.hpp"

#define __RBF_DEFAULT_POLYORDER 4
#define __RBF_DEFAULT_PARAM 0.1

namespace ablate::radialBasisV2 {

enum class RBFType { mq, phs, imq, ga };

class RBF : public ablate::solver::Solver {

  private:

    // Radial Basis Function type and parameters
    const ablate::radialBasisV2::RBFType rbfType = ablate::radialBasisV2::RBFType::mq;
    const PetscInt polyOrder = 4;
    const PetscReal rbfParam = 0.1;

    PetscReal (*RBFVal)(const PetscInt dim, PetscReal x[], PetscReal y[], const PetscReal param);
    PetscReal (*RBFDer)(const PetscInt dim, PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz, const PetscReal param);

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

  public:

    RBF(
      std::string solverId,
      std::shared_ptr<ablate::domain::Region>,
      std::shared_ptr<ablate::parameters::Parameters> options,
      ablate::radialBasisV2::RBFType rbfType,
      PetscInt rbfOrder,
      PetscReal rbfParam);


    /** SubDomain Register and Setup **/
    void Setup() override;
    void Initialize() override;

    // Derivative stuff
    void SetDerivatives(PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[], PetscBool useVertices);
    void SetDerivatives(PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[]);
    PetscReal EvalDer(ablate::domain::Field field, PetscInt c, PetscInt dx, PetscInt dy, PetscInt dz);  // Evaluate a derivative
    void SetupDerivativeStencils();   // Setup all derivative stencils. Useful if someone wants to remove setup cost when testing

    // Interpolation stuff
    void SetInterpolation(PetscBool hasInterpolation);
    PetscReal Interpolate(ablate::domain::Field field, PetscInt c, PetscReal xEval[3]);



};





std::istream& operator>>(std::istream& is, ablate::radialBasisV2::RBFType& v);

}

#endif  // ABLATELIBRARY_LEVELSETSOLVER_HPP
