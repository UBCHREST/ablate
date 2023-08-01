
#include "riemannCommon.hpp"


void ExpansionShockCalculation(const PetscReal pstar, const PetscReal gamma, const PetscReal gamm1, const PetscReal gamp1, const PetscReal p0, const PetscReal p, const PetscReal a, const PetscReal rho, PetscReal *f0, PetscReal *f1) {

    if (pstar <= p)  // expansion wave equation from Toro
    {
        *f0 = ((2. * a) / gamm1) * (PetscPowReal((pstar + p0) / (p + p0), 0.5 * gamm1 / gamma) - 1.);
        *f1 = (a / (p + p0) / gamma) * PetscPowReal((pstar + p0) / (p + p0), -0.5 * gamp1 / gamma);
    } else  // shock equation from Toro
    {
        PetscReal A = 2 / gamp1 / rho;
        PetscReal B = gamm1 * (p + p0) / gamp1;
        PetscReal sqterm = sqrt(A / (pstar + p0 + B));
        *f0 = (pstar - p) * sqterm;
        *f1 = sqterm * (1.0 - (0.5 * (pstar - p) / (B + pstar + p0)));
    }
}


ablate::finiteVolume::fluxCalculator::Direction riemannDirection( const PetscReal pstar,
                    const PetscReal uL, const PetscReal aL, const PetscReal rhoL, const PetscReal p0L, const PetscReal pL, const PetscReal gammaL, const PetscReal fL,
                    const PetscReal uR, const PetscReal aR, const PetscReal rhoR, const PetscReal p0R, const PetscReal pR, const PetscReal gammaR, const PetscReal fR,
                    PetscReal *massFlux, PetscReal *p12) {

    PetscReal STLR, SHLR, A, pRatio, gamma, gamm1, gamp1, astar, uX;

    // Now, start backing out the rest of the info.
    PetscReal ustar = 0.5 * (uL + uR + fR - fL);
    if (ustar >= 0) {
        gamma = gammaL;
        gamm1 = gamma - 1.0;
        gamp1 = gamma + 1.0;
        if (pstar <= pL)  // left expansion
        {
            astar = aL * PetscPowReal((pstar + p0L) / (pL + p0L), (gamm1 / (2 * gamma)));
            STLR = ustar - astar;
            if (STLR >= 0)  // positive tail wave
            {
                SHLR = uL - aL;
                if (SHLR >= 0)  // positive head wave
                {
                    *massFlux = rhoL * uL;
                    *p12 = pL;
                    uX = uL;
                } else  // Eq. 4.56 negative head wave
                {
                    A = rhoL * PetscPowReal((2 / gamp1) + ((gamm1 * uL) / (gamp1 * aL)), (2 / gamm1));
                    uX = 2 / gamp1 * (aL + (gamm1 * uL / 2));
                    *massFlux = A * uX;
                    *p12 = (pL + p0L) * PetscPowReal((2 / gamp1) + ((gamm1 * uL) / (gamp1 * aL)), (2 * gamma / gamm1));
                }
            } else {
                pRatio = (pstar + p0L) / (pL + p0L);
                *massFlux = rhoL * PetscPowReal(pRatio, 1.0 / gamma) * ustar;
                *p12 = pstar;
                uX = ustar;
            }

        } else  // Left shock
        {
            A = sqrt((gamp1 * (pstar + p0L) / 2 / gamma / (pL + p0L)) + (gamm1 / 2 / gamma));
            STLR = uL - (aL * A);  // shock wave speed
            if (STLR >= 0) {
                *massFlux = rhoL * uL;
                *p12 = pL;
                uX = uL;
            } else  // negative wave speed
            {
                pRatio = (pstar + p0L) / (pL + p0L);
                *massFlux = rhoL * (pRatio + (gamm1 / gamp1)) / (gamm1 * pRatio / gamp1 + 1) * ustar;
                *p12 = pstar;
                uX = ustar;
            }
        }
    } else  // negative ustar
    {
        gamma = gammaR;
        gamm1 = gamma - 1.0;
        gamp1 = gamma + 1.0;
        if (pstar <= pR)  // right expansion
        {
            SHLR = uR + aR;
            if (SHLR >= 0)  // positive head wave
            {
                astar = aR * PetscPowReal((pstar + p0R) / (pR + p0R), (gamm1 / (2 * gamma)));
                STLR = ustar + astar;
                if (STLR >= 0)  // positive tail wave
                {
                    pRatio = (pstar + p0R) / (pR + p0R);
                    *massFlux = rhoR * PetscPowReal(pRatio, 1 / gamma) * ustar;
                    *p12 = pstar;
                    uX = ustar;
                } else  // Eq. 4.56 negative tail wave
                {
                    A = rhoR * PetscPowReal((2 / gamp1) - ((gamm1 * uR) / (gamp1 * aR)), (2 / gamm1));
                    uX = 2 / gamp1 * (-aR + (gamm1 * uR / 2));
                    *massFlux = A * uX;
                    *p12 = (pR + p0R) * PetscPowReal((2 / gamp1) - ((gamm1 * uR) / (gamp1 * aR)), (2 * gamma / gamm1));
                }
            } else  // negative head wave
            {
                *massFlux = rhoR * uR;
                *p12 = pR;
                uX = uR;
            }
        } else  // right shock
        {
            A = sqrt((gamp1 * (pstar + p0R) / 2 / gamma / (pR + p0R)) + (gamm1 / 2 / gamma));
            STLR = uR + (aR * A);  // shock wave speed
            if (STLR >= 0) {
                pRatio = (pstar + p0R) / (pR + p0R);
                *massFlux = rhoR * (pRatio + (gamm1 / gamp1)) / (gamm1 * pRatio / gamp1 + 1) * ustar;
                *p12 = pstar;
                uX = ustar;
            } else  // negative wave speed
            {
                *massFlux = rhoR * uR;
                *p12 = pR;
                uX = uR;
            }
        }
    }

    return (uX > 0 ? ablate::finiteVolume::fluxCalculator::LEFT : ablate::finiteVolume::fluxCalculator::RIGHT);
}
