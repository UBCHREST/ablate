#ifndef ABLATELIBRARY_LEVELSETSOLVER_HPP
#define ABLATELIBRARY_LEVELSETSOLVER_HPP

#include <petsc.h>
#include <string>
#include <vector>
#include "levelSetFields.hpp"
#include "domain/RBF/rbf.hpp"
#include "solver/solver.hpp"

namespace ablate::levelSet {


class LevelSetSolver : public ablate::solver::Solver {
  private:
////    const ablate::RBF::RBFType rbfType = ablate::RBF::RBFType::MQ;

//    void Reinitialize(TS ts, ablate::solver::Solver &solver);

//    PetscInt dim;


    // Internal curvature and normal calculations
    PetscReal Curvature1D(PetscInt c);
    PetscReal Curvature2D(PetscInt c);
    PetscReal Curvature3D(PetscInt c);
    void Normal1D(PetscInt c, PetscReal *n);
    void Normal2D(PetscInt c, PetscReal *n);
    void Normal3D(PetscInt c, PetscReal *n);


//    // Possible initial shapes
//    PetscReal Sphere(PetscReal pos[], PetscReal center[], PetscReal radius);
//    PetscReal Ellipse(PetscReal pos[], PetscReal center[], PetscReal radius);
//    PetscReal Star(PetscReal pos[], PetscReal center[]);

    // The RBF to be used for derivatives
    std::shared_ptr<ablate::domain::rbf::RBF> rbf = nullptr;

    // Pointers to the fields for future use.
    const ablate::domain::Field *lsField = NULL;
    const ablate::domain::Field *curvField = NULL;
    const ablate::domain::Field *normalField = NULL;


  public:

  /**
   * Determine a level set field at cell centers at gives the same VOF as the input vector
   * @param field - The VOF field
   * @param f - The vector containing the VOF data
   */
//    void Reinitialize(const ablate::domain::Field *field, Vec f);


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

    /**
    * Returns the volume-of-fluid, face area, and/or volume of a given cell.
    * @param c - The cell of interest
    * @param vof - The volume-of-fluid in the cell
    * @param area - The interface length(2D) or area(3D) in the cell
    * @param vol - The area(2D) or volume(3D) of the cell
    */
    void VOF(const PetscInt c, PetscReal *vof, PetscReal *area, PetscReal *vol);

};

}

#endif  // ABLATELIBRARY_LEVELSETSOLVER_HPP
