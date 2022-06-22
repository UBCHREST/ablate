#ifndef ABLATELIBRARY_DER_HPP
#define ABLATELIBRARY_DER_HPP

#include <petsc.h>
#include <string>
#include <vector>
#include "utilities/petscError.hpp"
#include "rbf.hpp"
#include "lsSupport.hpp"

namespace ablate::levelSet {



class DerCalculator {

  private:
    PetscInt nDer = 0;                      // Number of derivative stencils which are pre-computed
    PetscInt *dxyz;                         // The derivatives: dx*100 + dy*10 + dz*1
    PetscInt *nStencil = nullptr;           // Length of each stencil
    PetscInt **stencilList = nullptr;       // IDs of the points in the stencil
    PetscReal **stencilWeights = nullptr;   // Weights of the points in the stencil

    PetscReal EvalDer_Internal(Vec f, PetscInt der, PetscInt nDer, PetscInt nStencil, PetscInt lst[], PetscReal wt[]);
    void SetupDerivativeStencils(std::shared_ptr<RBF> rbf, PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[], PetscInt **nStencil, PetscInt ***stencilList, PetscReal ***stencilWeights);

  public:

    // Constructor
    DerCalculator(std::shared_ptr<RBF> rbf = nullptr, PetscInt nDer = 0, PetscInt dx[] = {}, PetscInt dy[] = {}, PetscInt dz[] = {});

    // Deconstructor
    ~DerCalculator();

    // Evaluator
    PetscReal EvalDer(Vec f, PetscInt c, PetscInt dx, PetscInt dy, PetscInt dz);
};


}  // namespace ablate::levelSet


#endif  // ABLATELIBRARY_DER_HPP
