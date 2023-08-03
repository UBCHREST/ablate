#ifndef ABLATELIBRARY_RIEMANNCOMMON_HPP
#define ABLATELIBRARY_RIEMANNCOMMON_HPP

#include "fluxCalculator.hpp"


namespace ablate::finiteVolume::fluxCalculator {
/*
 * Computes the flux by treating all surfaces as Riemann problems, different stiffened gas on left/right.
 * Reference Chang and Liou, JCP, 2007, Appendix B
 */
class RiemannSolver : public FluxCalculator {
   protected:
     /**
     * Solve the Riemann problem.
     * @param uL: velocity on the left cell center
     * @param aL: speed of sound on the left center cell
     * @param rhoL: density on the left cell center
     * @param p0L: reference pressure for stiffened gas on left (pass in from EOS). Will be zero if perfect gas or single gas
     * @param pL: pressure on the left cell center
     * @param gammaL: specific heat ratio for gas on left (pass in from EOS)
     * @param uR: velocity on the right cell center
     * @param aR: speed of sound on the right center cell
     * @param rhoR: density on the right cell center
     * @param p0R: reference pressure for stiffened gas on right (pass in from EOS). Will be zero if perfect gas or single gas
     * @param pR: pressure on the right cell center
     * @param gammaR: specific heat ratio for gas on right (pass in from EOS)
     * @param pstar0: initial guess at the pressure across contact surface
     * @param massFlux: mass flux across the cell face(?)
     * @param p12: interface pressure(?)
     */
     static Direction riemannSolver(const PetscReal uL, const PetscReal aL, const PetscReal rhoL, const PetscReal p0L, const PetscReal pL, const PetscReal gammaL,
                                                                  const PetscReal uR, const PetscReal aR, const PetscReal rhoR, const PetscReal p0R, const PetscReal pR, const PetscReal gammaR,
                                                                  const PetscReal pstar0, PetscReal *massFlux, PetscReal *p12);






};

}  // namespace ablate::finiteVolume::fluxCalculator


#endif  // ABLATELIBRARY_RIEMANNCOMMON_HPP
