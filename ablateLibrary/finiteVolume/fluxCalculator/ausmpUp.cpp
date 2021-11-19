#include "ausmpUp.hpp"

ablate::finiteVolume::fluxCalculator::AusmpUp::AusmpUp(double mInf) : mInf(mInf) {}

ablate::finiteVolume::fluxCalculator::Direction ablate::finiteVolume::fluxCalculator::AusmpUp::AusmpUpFunction(void* ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR,
                                                                                                               PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal* massFlux, PetscReal* p12) {
    // Compute the density at the interface
    PetscReal rho12 = (0.5) * (rhoL + rhoR);

    // compute the speed of sound at a12
    PetscReal a12 = 0.5 * (aL + aR);  // Simple average of aL and aR.  This can be replaced with eq. 30;

    // Compute the left and right mach numbers
    PetscReal mL = uL / a12;
    PetscReal mR = uR / a12;

    // compute mInf2
    double* mInf = (double*)ctx;
    PetscReal mInf2 = PetscSqr(*mInf);

    // Compute mBar2 (eq 70)
    PetscReal mBar2 = (PetscSqr(uL) + PetscSqr(uR)) / (2.0 * a12 * a12);
    PetscReal mO2 = PetscMin(1.0, PetscMax(mBar2, mInf2));
    PetscReal mO = PetscSqrtReal(mO2);
    PetscReal fa = mO * (2.0 - mO);

    // compute the mach number on the interface
    PetscReal m12 = M4Plus(mL) + M4Minus(mR) - (Kp / fa) * PetscMax(1.0 - (sigma * mBar2), 0) * (pR - pL) / (rho12 * a12 * a12);

    // store the mass flux;
    Direction direction;
    if (m12 > 0) {
        direction = LEFT;
        *massFlux = a12 * m12 * rhoL;
    } else {
        direction = RIGHT;
        *massFlux = a12 * m12 * rhoR;
    }

    // Pressure
    if (p12) {
        double p5Plus = P5Plus(mL, fa);
        double p5Minus = P5Minus(mR, fa);

        *p12 = p5Plus * pL + p5Minus * pR - Ku * p5Plus * p5Minus * (rhoL + rhoR) * fa * a12 * (uR - uL);
    }
    return direction;
}

PetscReal ablate::finiteVolume::fluxCalculator::AusmpUp::M1Plus(PetscReal m) { return 0.5 * (m + PetscAbs(m)); }

PetscReal ablate::finiteVolume::fluxCalculator::AusmpUp::M2Plus(PetscReal m) { return 0.25 * PetscSqr(m + 1); }

PetscReal ablate::finiteVolume::fluxCalculator::AusmpUp::M1Minus(PetscReal m) { return 0.5 * (m - PetscAbs(m)); }
PetscReal ablate::finiteVolume::fluxCalculator::AusmpUp::M2Minus(PetscReal m) { return -0.25 * PetscSqr(m - 1); }

PetscReal ablate::finiteVolume::fluxCalculator::AusmpUp::M4Plus(PetscReal m) {
    if (PetscAbs(m) >= 1.0) {
        return M1Plus(m);
    } else {
        return M2Plus(m) * (1.0 - 16.0 * beta * M2Minus(m));
    }
}
PetscReal ablate::finiteVolume::fluxCalculator::AusmpUp::M4Minus(PetscReal m) {
    if (PetscAbs(m) >= 1.0) {
        return M1Minus(m);
    } else {
        return M2Minus(m) * (1.0 + 16.0 * beta * M2Plus(m));
    }
}
PetscReal ablate::finiteVolume::fluxCalculator::AusmpUp::P5Plus(PetscReal m, double fa) {
    if (PetscAbs(m) >= 1.0) {
        return (M1Plus(m) / (m + 1E-30));
    } else {
        // compute alpha
        double alpha = 3.0 / 16.0 * (-4.0 + 5 * fa * fa);

        return (M2Plus(m) * ((2.0 - m) - 16. * alpha * m * M2Minus(m)));
    }
}
PetscReal ablate::finiteVolume::fluxCalculator::AusmpUp::P5Minus(PetscReal m, double fa) {
    if (PetscAbs(m) >= 1.0) {
        return (M1Minus(m) / (m + 1E-30));
    } else {
        double alpha = 3.0 / 16.0 * (-4.0 + 5 * fa * fa);
        return (M2Minus(m) * ((-2.0 - m) + 16. * alpha * m * M2Plus(m)));
    }
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::fluxCalculator::FluxCalculator, ablate::finiteVolume::fluxCalculator::AusmpUp, "A sequel to AUSM, Part II: AUSM+-up for all speeds, Meng-Sing Liou, Pages 137-170, 2006",
         ARG(double, "mInf", "the reference mach number"));