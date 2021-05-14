#include <petscsys.h>
#include "fluxDifferencer.h"

static PetscBool fluxDifferenceInitialized = PETSC_FALSE;
static PetscFunctionList fluxDifferencerList = NULL;

/* Computes the min/plus values..
 * sPm: minus split pressure (P-), Capital script P in reference
 * sMm: minus split Mach Number (M-), Capital script M in reference
 * sPp: plus split pressure (P+), Capital script P in reference
 * sMp: plus split Mach Number (M+), Capital script M in reference
 */
static void AusmFluxSplitCalculator(PetscReal Mm, PetscReal* sPm, PetscReal* sMm,
                                    PetscReal Mp, PetscReal* sPp, PetscReal *sMp) {

    if (PetscAbsReal(Mm) <= 1.) {
        *sMm = -0.25 * PetscSqr(Mm - 1);
        *sPm = -(*sMm) * (2 + Mm);
    }else {
        *sMm = 0.5 * (Mm - PetscAbsReal(Mm));
        *sPm = (*sMm) / Mm;
    }
    if (PetscAbsReal(Mp) <= 1.) {
        *sMp = 0.25 * PetscSqr(Mp + 1);
        *sPp = (*sMp) * (2 - Mp);
    }else {
        *sMp = 0.5 * (Mp + PetscAbsReal(Mp));
        *sPp = (*sMp) / Mp;
    }

    // compute the combined M
    PetscReal m = *sMm + *sMp;

    if (m < 0){
        // M- on Right
        *sMm = m;
        *sMp = 0.0;//Zero out the left contribution
    }else{
        // M+ on Left
        *sMm = 0.0;
        *sMp = m;//Zero out the right contribution
    }
}

/* Produces a split Mach and pressure that reproduces the average face value.
*/
static void AverageSplitCalculator(PetscReal Mm, PetscReal* sPm, PetscReal* sMm,
                                    PetscReal Mp, PetscReal* sPp, PetscReal *sMp) {
    *sPm = 0.5;
    *sPp = 0.5;
    *sMm = Mm/2.0;
    *sMp = Mp/2.0;
}

/* Turns off the values resulting no flux.  This is useful for debug and testinjg
*/
static void OffSplitCalculator(PetscReal Mm, PetscReal* sPm, PetscReal* sMm,
                                   PetscReal Mp, PetscReal* sPp, PetscReal *sMp) {
    *sPm = 0.0;
    *sPp = 0.0;
    *sMm = 0.0;
    *sMp = 0.0;
}


/*
 * Returns the plus split Mach number (+) using Van Leer splitting
 * - Reference 1: "A New Flux Splitting Scheme" Liou and Steffen, pg 26, Eqn (6), 1993
 * - Reference 2: "A Sequel to AUSM: AUSM+" Liou, pg 366, Eqn (8), 1996, actually eq. 19a
 * - Reference 3: "A sequel to AUSM, Part II: AUSM+-up for all speeds" Liou, pg 141, Eqn (18), 2006
 * - Capital script M_(1) in this reference
 */
static PetscReal sM1p (PetscReal M) {
    // Equation: 1/2*[M+|M|]
    return (0.5*(M+PetscAbsReal(M)));
}

/*
 * Returns the minus split Mach number (-) using Van Leer splitting
 * - Reference 1: "A New Flux Splitting Scheme" Liou and Steffen, pg 26, Eqn (6), 1993
 * - Reference 2: "A Sequel to AUSM: AUSM+" Liou, pg 366, Eqn (8), 1996
 * - Reference 3: "A sequel to AUSM, Part II: AUSM+-up for all speeds" Liou, pg 141, Eqn (18), 2006
 * - Capital script M_(1) in this reference
 */
static PetscReal sM1m (PetscReal M) {
    // Equation: 1/2*[M-|M|]
    return (0.5*(M-PetscAbsReal(M)));
}

// Parameters alpha and beta
// - Give improved results over AUSM and results are comparable to Roe splitting
// - Reference: "A Sequel to AUSM: AUSM+", Liou, pg 368, Eqn (22a, 22b), 1996
const static PetscReal AUSMbeta  = 1.e+0 / 8.e+0;
const static PetscReal AUSMalpha = 3.e+0 / 16.e+0;


static PetscErrorCode FluxDifferencerInitialize(){
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    if (!fluxDifferenceInitialized){
        fluxDifferenceInitialized = PETSC_TRUE;
        ierr = FluxDifferencerRegister("ausm", AusmFluxSplitCalculator);CHKERRQ(ierr);
        ierr = FluxDifferencerRegister("average", AverageSplitCalculator);CHKERRQ(ierr);
        ierr = FluxDifferencerRegister("off", OffSplitCalculator);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

PetscErrorCode FluxDifferencerRegister(const char * name, const FluxDifferencerFunction function) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    if (!fluxDifferenceInitialized) {
        ierr = FluxDifferencerInitialize();CHKERRQ(ierr);
    }
    ierr = PetscFunctionListAdd(&fluxDifferencerList,name,function);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode FluxDifferencerGet(const char* name, FluxDifferencerFunction* function){
    PetscFunctionBeginUser;
    PetscErrorCode ierr = FluxDifferencerInitialize();CHKERRQ(ierr);
    ierr = PetscFunctionListFind(fluxDifferencerList,name,function);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode FluxDifferencerListGet(PetscFunctionList* list){
    PetscFunctionBeginUser;
    *list = fluxDifferencerList;
    PetscFunctionReturn(0);
}