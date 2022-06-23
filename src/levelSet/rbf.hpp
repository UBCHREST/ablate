#ifndef ABLATELIBRARY_RBF_HPP
#define ABLATELIBRARY_RBF_HPP


#include <string>
#include <vector>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscksp.h>
#include "domain/domain.hpp"

namespace ablate::levelSet {

class RBF {

  private:
    PetscInt  dim = -1;         // Dimension of the DM
    PetscInt  p = -1;           // The supplementary polynomial order
    PetscInt  nPoly = -1;       // The number of polynomial components to include
    DM        dm = nullptr;     // For now just use the entire DM. When this is moved over to the Domain/Subdomain class this will be modified.

    void Matrix(PetscInt c, PetscInt nCells, PetscInt list[], PetscReal x[], Mat *LUA);

  protected:
    PetscReal DistanceSquared(PetscReal x[], PetscReal y[]);
    PetscReal DistanceSquared(PetscReal x[]);

  public:

    // These will be overwritten in the derived classes
    virtual PetscReal Val(PetscReal x[], PetscReal y[]) = 0;
    virtual PetscReal Der(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) = 0;

    // Constructor
    RBF(DM dm = nullptr, PetscInt p = -1);

    // Print all of the parameters.
    void ShowParameters();

    // Return the mesh associated with the RBF
    inline DM& GetDM() noexcept { return dm; }

    inline PetscInt GetNPoly() { return nPoly; }


    // The finite difference weights for derivatives
    void Weights(PetscInt c, PetscInt nCells, PetscInt list[], PetscInt nDer, PetscInt dx[], PetscInt dy[], PetscInt dz[], PetscReal *weights[]);


};

class PHS: public RBF {
  private:
    PetscInt  phsOrder = -1;    // The PHS order

    PetscReal InternalVal(PetscReal x[], PetscReal y[]);
    PetscReal InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz);
  public:
    PHS(DM dm = nullptr, PetscInt p = -1, PetscInt m = -1);
    PetscReal Val(PetscReal x[], PetscReal y[]) override {return InternalVal(std::move(x), std::move(y)); }
    PetscReal Der(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override {return InternalDer(std::move(x), std::move(dx), std::move(dy), std::move(dz)); }
};

class MQ: public RBF {
  private:
    PetscReal scale = -1;

    PetscReal InternalVal(PetscReal x[], PetscReal y[]);
    PetscReal InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz);
  public:
    MQ(DM dm = nullptr, PetscInt p = -1, PetscReal scale = -1);
    PetscReal Val(PetscReal x[], PetscReal y[]) override {return InternalVal(std::move(x), std::move(y)); }
    PetscReal Der(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override {return InternalDer(std::move(x), std::move(dx), std::move(dy), std::move(dz)); }
};

class IMQ: public RBF {
  private:
    PetscReal scale = -1;

    PetscReal InternalVal(PetscReal x[], PetscReal y[]);
    PetscReal InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz);
  public:
    IMQ(DM dm = nullptr, PetscInt p = -1, PetscReal scale = -1);
    PetscReal Val(PetscReal x[], PetscReal y[]) override {return InternalVal(std::move(x), std::move(y)); }
    PetscReal Der(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override {return InternalDer(std::move(x), std::move(dx), std::move(dy), std::move(dz)); }
};

class GA: public RBF {
  private:
    PetscReal scale = -1;

    PetscReal InternalVal(PetscReal x[], PetscReal y[]);
    PetscReal InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz);
  public:
    GA(DM dm = nullptr, PetscInt p = -1, PetscReal scale = -1);
    PetscReal Val(PetscReal x[], PetscReal y[]) override {return InternalVal(std::move(x), std::move(y)); }
    PetscReal Der(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override {return InternalDer(std::move(x), std::move(dx), std::move(dy), std::move(dz)); }
};

}  // namespace ablate::levelSet

#endif  // ABLATELIBRARY_RBF_HPP

