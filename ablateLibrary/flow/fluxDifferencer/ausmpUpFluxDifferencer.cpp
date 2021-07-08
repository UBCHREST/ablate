#include "ausmpUpFluxDifferencer.hpp"



// A sequel to AUSM, Part II: AUSM+-up for all speeds
void ablate::flow::fluxDifferencer::AusmpUpFluxDifferencer::AusmpUpFluxDifferencerFunction(void*, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL,
                                                                                           PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR,
                                                                                           PetscReal * massFlux, PetscReal *p12) {
    // Compute teh density at one half
    PetscReal rho12 = (0.5) * (rhoL + rhoR);

    // compute the speed of sound at a12
    PetscReal a12 = 0.5 * (aL + aR);  // Simple average of aL and aR.  This can be replaced with eq. 30;

    // Compute the left and right mach numbers
    PetscReal mL = uL / a12;
    PetscReal mR = uR / a12;

    // compute mInf2
    PetscReal mInf2 = 1E-8;

    // Compute mBar2 (eq 70)
    PetscReal mBar2 = (PetscSqr(uL) + PetscSqr(uR)) / (2.0 * a12 * a12);
    PetscReal mO2 = PetscMin(1.0, PetscMax(mBar2, mInf2));
    PetscReal mO = PetscSqrtReal(mO2);
    PetscReal fa = mO * (2.0 - mO);

    // compute the mach number on the interface
    PetscReal m12 = M4Plus(mL) + M4Minus(mL) - Kp / (fa)*PetscMax(1.0 - sigma * mBar2, 0) * (pR - pL) / (rho12 * a12 * a12);

    // store the mass flux;
    *massFlux = a12 * m12 * (m12 > 0 ? rhoL : rhoR);

    // Pressure
    if (p12) {
        double p5Plus = P5Plus(mL);
        double p5Minus = P5Minus(mR);

        *p12 = p5Plus * pL + p5Minus * pR - Ku * p5Plus * p5Minus + (rhoL + rhoR) * fa * a12 * (uR - uL);
    }
}

PetscReal ablate::flow::fluxDifferencer::AusmpUpFluxDifferencer::M1Plus(PetscReal m) {
    return 0.5*(m + PetscAbs(m));
}

PetscReal ablate::flow::fluxDifferencer::AusmpUpFluxDifferencer::M2Plus(PetscReal m) {
    return 0.25* PetscSqr(m + 1);
}

PetscReal ablate::flow::fluxDifferencer::AusmpUpFluxDifferencer::M1Minus(PetscReal m) {
    return 0.5*(m - PetscAbs(m));
}
PetscReal ablate::flow::fluxDifferencer::AusmpUpFluxDifferencer::M2Minus(PetscReal m) {
    return -0.25* PetscSqr(m - 1);
}

PetscReal ablate::flow::fluxDifferencer::AusmpUpFluxDifferencer::M4Plus(PetscReal m) {
    if(PetscAbs(m) >= 1.0){
        return M1Plus(m);
    }else{
        return M2Plus(m)*(1.0 - 16.0*beta* M2Minus(m));
    }
}
PetscReal ablate::flow::fluxDifferencer::AusmpUpFluxDifferencer::M4Minus(PetscReal m) {
    if(PetscAbs(m) >= 1.0){
        return M1Minus(m);
    }else{
        return M2Minus(m)*(1.0 + 16.0*beta* M2Plus(m));
    }}
PetscReal ablate::flow::fluxDifferencer::AusmpUpFluxDifferencer::P5Plus(PetscReal m) {
    if (PetscAbs(m) >= 1.0) {
        return (M1Plus(m)/(m + 1E-30));
    }
    else {
        return (M2Plus(m)*((2.0-m) - 16.*alpha*m*M2Minus(m)));
    }
}
PetscReal ablate::flow::fluxDifferencer::AusmpUpFluxDifferencer::P5Minus(PetscReal m) {
    if (PetscAbs(m) >= 1.0) {
        return (M1Minus(m)/(m + 1E-30));
    }
    else {
        return (M2Minus(m)*((-2.0-m) + 16.*alpha*m*M2Plus(m)));
    }
}
