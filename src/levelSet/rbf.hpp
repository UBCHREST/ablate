#ifndef ABLATELIBRARY_RBF_HPP
#define ABLATELIBRARY_RBF_HPP


#include <string>
#include <vector>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscksp.h>
#include "domain/domain.hpp"
#include "lsSupport.hpp"


namespace ablate::levelSet {

class RBF {

  private:
    PetscInt  dim = -1;                 // Dimension of the DM
    PetscInt  p = -1;                   // The supplementary polynomial order
    PetscInt  nPoly = -1;               // The number of polynomial components to include
    DM        dm = nullptr;             // For now just use the entire DM. When this is moved over to the Domain/Subdomain class this will be modified.
    PetscInt  minNumberCells = -1;      // Minimum number of cells needed to compute the RBF
    PetscBool useVertices = PETSC_TRUE; // Use vertices or edges/faces when computing neighbor cells

    // Compute the LU-decomposition of the augmented RBF matrix given a cell list.
    void Matrix(PetscInt c, PetscInt nCells, PetscInt list[], PetscReal x[], Mat *LUA);



    // The finite difference weights for derivatives
    void Weights(PetscInt c, PetscInt nCells, PetscInt list[], PetscInt nDer, PetscInt dxyz[], PetscReal *weights[]);



    // Derivative data
    PetscBool hasDerivativeInformation = PETSC_FALSE;
    PetscInt nDer = 0;                      // Number of derivative stencils which are pre-computed
    PetscInt *dxyz = nullptr;
    PetscInt *nStencil = nullptr;           // Length of each stencil
    PetscInt **stencilList = nullptr;       // IDs of the points in the stencil
    PetscReal **stencilWeights = nullptr;   // Weights of the points in the stencil

    // Setup the derivative stencil at a point. There is no need for anyone outside of RBF to call this
    void SetupDerivativeStencils(PetscInt c);

    PetscBool forInterpolation = PETSC_FALSE;



  protected:
    PetscReal DistanceSquared(PetscReal x[], PetscReal y[]);
    PetscReal DistanceSquared(PetscReal x[]);

    // These will be overwritten in the derived classes
    virtual PetscReal RBFVal(PetscReal x[], PetscReal y[]) = 0;
    virtual PetscReal RBFDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) = 0;

  public:


    // Constructor
    RBF(DM dm = nullptr, PetscInt p = -1);

    // Desctructor
    ~RBF();

    // Print all of the parameters.
    void ShowParameters();

    // Return the mesh associated with the RBF
    inline DM& GetDM() noexcept { return dm; }

    inline PetscInt GetNPoly() { return nPoly; }

    // Setup all derivative stencils. Useful if someone wants to remove setup cost when testing
    void SetupDerivativeStencils();

    // Derivative stuff
    void SetDerivatives(PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[], PetscBool useVertices);
    void SetDerivatives(PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[]);

    PetscReal EvalDer(Vec f, PetscInt c, PetscInt dx, PetscInt dy, PetscInt dz);



};

class PHS: public RBF {
  private:
    PetscInt  phsOrder = -1;    // The PHS order

    PetscReal InternalVal(PetscReal x[], PetscReal y[]);
    PetscReal InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz);
  public:
    PHS(DM dm = nullptr, PetscInt p = -1, PetscInt m = -1);
    PetscReal RBFVal(PetscReal x[], PetscReal y[]) override {return InternalVal(std::move(x), std::move(y)); }
    PetscReal RBFDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override {return InternalDer(std::move(x), std::move(dx), std::move(dy), std::move(dz)); }
};

class MQ: public RBF {
  private:
    PetscReal scale = -1;

    PetscReal InternalVal(PetscReal x[], PetscReal y[]);
    PetscReal InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz);
  public:
    MQ(DM dm = nullptr, PetscInt p = -1, PetscReal scale = -1);
    PetscReal RBFVal(PetscReal x[], PetscReal y[]) override {return InternalVal(std::move(x), std::move(y)); }
    PetscReal RBFDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override {return InternalDer(std::move(x), std::move(dx), std::move(dy), std::move(dz)); }
};

class IMQ: public RBF {
  private:
    PetscReal scale = -1;

    PetscReal InternalVal(PetscReal x[], PetscReal y[]);
    PetscReal InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz);
  public:
    IMQ(DM dm = nullptr, PetscInt p = -1, PetscReal scale = -1);
    PetscReal RBFVal(PetscReal x[], PetscReal y[]) override {return InternalVal(std::move(x), std::move(y)); }
    PetscReal RBFDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override {return InternalDer(std::move(x), std::move(dx), std::move(dy), std::move(dz)); }
};

class GA: public RBF {
  private:
    PetscReal scale = -1;

    PetscReal InternalVal(PetscReal x[], PetscReal y[]);
    PetscReal InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz);
  public:
    GA(DM dm = nullptr, PetscInt p = -1, PetscReal scale = -1);
    PetscReal RBFVal(PetscReal x[], PetscReal y[]) override {return InternalVal(std::move(x), std::move(y)); }
    PetscReal RBFDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override {return InternalDer(std::move(x), std::move(dx), std::move(dy), std::move(dz)); }
};

}  // namespace ablate::levelSet

#endif  // ABLATELIBRARY_RBF_HPP

