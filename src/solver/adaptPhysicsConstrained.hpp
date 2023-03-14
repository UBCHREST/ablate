#ifndef ABLATELIBRARY_ADAPTPHYSICSCONSTRAINED_HPP
#define ABLATELIBRARY_ADAPTPHYSICSCONSTRAINED_HPP

#include <petsc.h>
#include <petsc/private/tsimpl.h>
#include "utilities/petscUtilities.hpp"
namespace ablate::solver {

/**
 * This is a static class that holds the required code needed by petsc for the adapt physics constrained time step constraint.
 * This is just a slight modification for the adapt basic in petsc
 */
class AdaptPhysicsConstrained {
   private:
    static inline const char name[] = "physicsConstrained";

    /**
     * The physics based implementation of TSAdaptChoose
     * @param adapt
     * @param ts
     * @param h
     * @param next_sc
     * @param next_h
     * @param accept
     * @param wlte
     * @param wltea
     * @param wlter
     * @return
     */
    static PetscErrorCode TSAdaptChoose(TSAdapt adapt, TS ts, PetscReal h, PetscInt *next_sc, PetscReal *next_h, PetscBool *accept, PetscReal *wlte, PetscReal *wltea, PetscReal *wlter);

    /**
     * Static call to modify the adapt petsc object into an adapt physics implementation
     * @param adapt
     * @return
     */
    static PetscErrorCode TSAdaptCreate(TSAdapt adapt);

    /**
     * Compute the initial time step based upon physics
     * @param adapt
     * @return
     */
    static void AdaptInitializer(TS ts, TSAdapt adapt);

   public:
    /**
     * Function to register the ts adapt with petsc
     */
    static void Register();

    /**
     * Prevent this class from being used in an non static way
     */
    AdaptPhysicsConstrained() = delete;
};
}  // namespace ablate::solver

#endif  // ABLATELIBRARY_ADAPTPHYSICS_HPP
