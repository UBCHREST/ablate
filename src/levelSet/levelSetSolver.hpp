#ifndef ABLATELIBRARY_LEVELSETSOLVER_HPP
#define ABLATELIBRARY_LEVELSETSOLVER_HPP

#include <petsc.h>
#include <string>
#include <vector>
#include "levelSetFields.hpp"
#include "domain/RBF/rbf.hpp"

namespace ablate::levelSet {


class LevelSetSolver : public ablate::solver::Solver {
//  private:
////    const ablate::RBF::RBFType rbfType = ablate::RBF::RBFType::MQ;

//    void Reinitialize(TS ts, ablate::solver::Solver &solver);

//    PetscInt dim;


    // Internal curvature and normal calculations
    PetscReal Curvature2D(PetscInt c);
    PetscReal Curvature3D(PetscInt c);
    void Normal2D(PetscInt c, PetscReal *n);
    void Normal3D(PetscInt c, PetscReal *n);


//    // Possible initial shapes
//    PetscReal Sphere(PetscReal pos[], PetscReal center[], PetscReal radius);
//    PetscReal Ellipse(PetscReal pos[], PetscReal center[], PetscReal radius);
//    PetscReal Star(PetscReal pos[], PetscReal center[]);





//    // Copied from current ABLATE code. Need to talk to Matt M. about how to integrate
//    const std::shared_ptr<domain::Region> region = nullptr;

    // The RBF to be used for derivatives
    std::shared_ptr<ablate::domain::rbf::RBF> rbf = nullptr;


//    // The level set data
//    Vec phi = nullptr;

//    // The curvature
//    Vec curv = nullptr;

//    // The unit normal
//    Vec normal = nullptr;

//    // Underlying mesh
//    DM dm = nullptr;

//    PetscInt dim;

  // Pointers to the fields for future use.
  const ablate::domain::Field *lsField = NULL;
  const ablate::domain::Field *curvField = NULL;
  const ablate::domain::Field *normalField = NULL;


  public:

//    void VOF(const PetscInt p, PetscReal *vof, PetscReal *area, PetscReal *vol);
//    enum class levelSetShape {SPHERE, ELLIPSE, STAR};

//    void Advect(Vec vel, const PetscReal dt);

//    bool HasInterface(const PetscInt p);

//    void Reinitialize(Vec VOF);


//    // Level set function interpolation
//    PetscReal Interpolate(const PetscReal x, const double y, const double z);
//    PetscReal Interpolate(PetscReal xyz[3]);



    LevelSetSolver(
      std::string solverId,
      std::shared_ptr<ablate::domain::Region>,
      std::shared_ptr<ablate::parameters::Parameters> options,
      const std::shared_ptr<ablate::domain::rbf::RBF>& rbf);


    /** SubDomain Register and Setup **/
    void Setup() override;
    void Initialize() override;


    // Public curvature and normal functions
    PetscReal Curvature(PetscInt c);
    void Normal(PetscInt c, PetscReal *n);
    void ComputeAllNormal();
    void ComputeAllCurvature();

    // Returns the volume-of-fluid, face area, and/or volume of a given cell.
    void VOF(const PetscInt c, PetscReal *vof, PetscReal *area, PetscReal *vol);


////    std::string GetRBFType();


};

}

#endif  // ABLATELIBRARY_LEVELSETSOLVER_HPP
