#include "riemann2Gas.hpp"
#include <eos/perfectGas.hpp>
ablate::finiteVolume::fluxCalculator::Direction ablate::finiteVolume::fluxCalculator::Riemann2Gas::Riemann2GasFluxFunction(void *ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL,
                                                                                                                           PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal pgsAlpha,
                                                                                                                           PetscReal *massFlux,

                                                                                                                           PetscReal *p12) {
    /*
     * gammaL: specific heat ratio for gas on left (pass in from EOS)
     * gamLm1 = gammaL - 1
     * gamLp1 = gammaL + 1
     * gammaR: specific heat ratio for gas on right (pass in from EOS)
     * gamRm1 = gammaR - 1
     * gamRp1 = gammaR + 1
     * uL: velocity on the left cell center
     * uR: velocity on the right cell center
     * rhoR: density on the right cell center
     * rhoL: density on the left cell center
     * pR: pressure on the right cell center
     * pL: pressure on the left cell center
     * aR: SoS on the  right center cell
     * aL: SoS on the left center cell
     * pstar: pressure across contact surface
     * ustar: velocity across contact surface
     * rhostarR: density on the right of the contact surface
     * whostarL: density on the left of the contact surface
     * err: final residual for iteration
     * MAXIT: maximum iteration times
     */
    PetscInt i = 0;
    const PetscInt MAXIT = 100;
    const PetscReal err = 1e-6;
    auto gammaVec = (PetscReal *)ctx;  // pass-in specific heat ratio from EOS_Left
    // This is where Riemann solver lives.
    PetscReal gammaL = gammaVec[0];
    PetscReal gamLm1 = gammaL - 1.0, gamLp1 = gammaL + 1.0;
    PetscReal gammaR = gammaVec[1];
    PetscReal gamRm1 = gammaR - 1.0, gamRp1 = gammaR + 1.0;
    PetscReal gamma, gamm1, gamp1;

    PetscReal pold, pstar, ustar, f_L_0, f_L_1, f_R_0, f_R_1, del_u = uR - uL;
    PetscReal A, B, sqterm;
    PetscReal astar, STLR, SHLR, uX;

    // Here is the initial guess for pstar - assuming two exapansion wave (need to change for 2 gasses)
    pstar = aL + aR - (0.5 * gamLm1 * (uR - uL));
    pstar = pstar / ((aL / PetscPowReal(pL, 0.5 * gamLm1 / gammaL)) + (aR / PetscPowReal(pR, 0.5 * gamRm1 / gammaR)));
    pstar = PetscPowReal(pstar, 2.0 * gammaL / gamLm1);
    pstar = 0.5 * (pR + pL);

    if (pstar <= pL)  // expansion wave equation from Toro
    {
        gamma = gammaL;
        gamm1 = gamLm1;
        gamp1 = gamLp1;
        f_L_0 = ((2. * aL) / gamm1) * (PetscPowReal(pstar / pL, 0.5 * gamm1 / gamma) - 1.);
        f_L_1 = (aL / pL / gamma) * PetscPowReal(pstar / pL, -0.5 * gamp1 / gamma);
    } else  // shock equation from Toro
    {
        gamma = gammaL;
        gamm1 = gamLm1;
        gamp1 = gamLp1;
        A = 2 / gamp1 / rhoL;
        B = gamm1 * pL / gamp1;
        sqterm = sqrt(A / (pstar + B));
        f_L_0 = (pstar - pL) * sqterm;
        f_L_1 = sqterm * (1.0 - (0.5 * (pstar - pL) / (B + pstar)));
    }
    if (pstar <= pR)  // expansion wave equation from Toro
    {
        gamma = gammaR;
        gamm1 = gamRm1;
        gamp1 = gamRp1;
        f_R_0 = ((2 * aR) / gamm1) * (PetscPowReal(pstar / pR, 0.5 * gamm1 / gamma) - 1);
        f_R_1 = (aR / pR / gamma) * PetscPowReal(pstar / pR, -0.5 * gamp1 / gamma);
    } else  // shock equation from Toro
    {
        gamma = gammaR;
        gamm1 = gamRm1;
        gamp1 = gamRp1;
        A = 2 / gamp1 / rhoR;
        B = gamm1 * pR / gamp1;
        sqterm = sqrt(A / (pstar + B));
        f_R_0 = (pstar - pR) * sqterm;
        f_R_1 = sqterm * (1.0 - (0.5 * (pstar - pR) / (B + pstar)));
    }

    // iteration starts
    while (PetscAbsReal(f_L_0 + f_R_0 + del_u) > err && i <= MAXIT)  // Newton's method
    {
        pold = pstar;
        pstar = pold - (f_L_0 + f_R_0 + del_u) / (f_L_1 + f_R_1);  // new guess
        if (pstar <= pL)                                           // expansion wave equation from Toto
        {
            gamma = gammaL;
            gamm1 = gamLm1;
            gamp1 = gamLp1;
            f_L_0 = ((2. * aL) / gamm1) * (PetscPowReal(pstar / pL, 0.5 * gamm1 / gamma) - 1.);
            f_L_1 = (aL / pL / gamma) * PetscPowReal(pstar / pL, -0.5 * gamp1 / gamma);
        } else  // shock equation from Toro
        {
            gamma = gammaL;
            gamm1 = gamLm1;
            gamp1 = gamLp1;
            A = 2 / gamp1 / rhoL;
            B = gamm1 * pL / gamp1;
            sqterm = sqrt(A / (pstar + B));
            f_L_0 = (pstar - pL) * sqterm;
            f_L_1 = sqterm * (1.0 - (0.5 * (pstar - pL) / (B + pstar)));
        }
        if (pstar <= pR)  // expansion wave equation from Toto
        {
            gamma = gammaR;
            gamm1 = gamRm1;
            gamp1 = gamRp1;
            f_R_0 = ((2 * aR) / gamm1) * (PetscPowReal(pstar / pR, 0.5 * gamm1 / gamma) - 1);
            f_R_1 = (aR / pR / gamma) * PetscPowReal(pstar / pR, -0.5 * gamp1 / gamma);
        } else  // shock equation from Toro
        {
            gamma = gammaR;
            gamm1 = gamRm1;
            gamp1 = gamRp1;
            A = 2 / gamp1 / rhoR;
            B = gamm1 * pR / gamp1;
            sqterm = sqrt(A / (pstar + B));
            f_R_0 = (pstar - pR) * sqterm;
            f_R_1 = sqterm * (1.0 - (0.5 * (pstar - pR) / (B + pstar)));
        }
        i++;
    }
    if (i > MAXIT) {
        throw std::runtime_error("Can't find pstar; Iteration not converging; Go back and do it again");
    }

    // Now, start backing out the rest of the info.
    ustar = 0.5 * (uL + uR + f_R_0 - f_L_0);
    if (ustar >= 0) {
        if (pstar <= pL)  // left expansion
        {
            gamma = gammaL;
            gamm1 = gamLm1;
            gamp1 = gamLp1;
            astar = aL * PetscPowReal(pstar / pL, (gamm1 / (2 * gamma)));
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
                    *p12 = pL * PetscPowReal((2 / gamp1) + ((gamm1 * uL) / (gamp1 * aL)), (2 * gamma / gamm1));
                }
            } else {
                gamma = gammaL;
                gamm1 = gamLm1;
                gamp1 = gamLp1;
                auto pRatio = pstar / pL;
                *massFlux = rhoL * PetscPowReal(pRatio, 1.0 / gamma) * ustar;
                *p12 = pstar;
                uX = ustar;
            }

        } else  // Left shock
        {
            gamma = gammaL;
            gamm1 = gamLm1;
            gamp1 = gamLp1;
            A = sqrt((gamp1 * pstar / 2 / gamma / pL) + (gamm1 / 2 / gamma));
            STLR = uL - (aL * A);  // shock wave speed
            if (STLR >= 0) {
                *massFlux = rhoL * uL;
                *p12 = pL;
                uX = uL;
            } else  // negative wave speed
            {
                gamma = gammaL;
                gamm1 = gamLm1;
                gamp1 = gamLp1;
                auto pRatio = pstar / pL;
                *massFlux = rhoL * (pRatio + (gamm1 / gamp1)) / (gamm1 * pRatio / gamp1 + 1) * ustar;
                *p12 = pstar;
                uX = ustar;
            }
        }
    } else  // negative ustar
    {
        if (pstar <= pR)  // right expansion
        {
            SHLR = uR + aR;
            if (SHLR >= 0)  // positive head wave
            {
                gamma = gammaR;
                gamm1 = gamRm1;
                gamp1 = gamRp1;
                astar = aR * PetscPowReal(pstar / pR, (gamm1 / (2 * gamma)));
                STLR = ustar + astar;
                if (STLR >= 0)  // positive tail wave
                {
                    auto pRatio = pstar / pR;
                    *massFlux = rhoR * PetscPowReal(pRatio, 1 / gamma) * ustar;
                    *p12 = pstar;
                    uX = ustar;
                } else  // Eq. 4.56 negative tail wave
                {
                    A = rhoR * PetscPowReal((2 / gamp1) - ((gamm1 * uR) / (gamp1 * aR)), (2 / gamm1));
                    uX = 2 / gamp1 * (-aR + (gamm1 * uR / 2));
                    *massFlux = A * uX;
                    *p12 = pR * PetscPowReal((2 / gamp1) - ((gamm1 * uR) / (gamp1 * aR)), (2 * gamma / gamm1));
                }
            } else  // negative head wave
            {
                *massFlux = rhoR * uR;
                *p12 = pR;
                uX = uR;
            }
        } else  // right shock
        {
            gamma = gammaR;
            gamm1 = gamRm1;
            gamp1 = gamRp1;
            A = sqrt((gamp1 * pstar / 2 / gamma / pR) + (gamm1 / 2 / gamma));
            STLR = uR + (aR * A);  // shock wave speed
            if (STLR >= 0) {
                auto pRatio = pstar / pR;
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

    return uX > 0 ? LEFT : RIGHT;
}
ablate::finiteVolume::fluxCalculator::Riemann2Gas::Riemann2Gas(std::shared_ptr<eos::EOS> eosL, std::shared_ptr<eos::EOS> eosR) {
    auto perfectGasEosL = std::dynamic_pointer_cast<eos::PerfectGas>(eosL);
    auto perfectGasEosR = std::dynamic_pointer_cast<eos::PerfectGas>(eosR);
    if (!perfectGasEosL) {
        throw std::invalid_argument("ablate::flow::fluxCalculator::Direction ablate::flow::fluxCalculator::Riemann2Gas left only accepts EOS of type eos::PerfectGas");
    }
    if (!perfectGasEosR) {
        throw std::invalid_argument("ablate::flow::fluxCalculator::Direction ablate::flow::fluxCalculator::Riemann2Gas right only accepts EOS of type eos::PerfectGas");
    }
    gammaVec[0] = perfectGasEosL->GetSpecificHeatRatio();
    gammaVec[1] = perfectGasEosR->GetSpecificHeatRatio();
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::fluxCalculator::FluxCalculator, ablate::finiteVolume::fluxCalculator::Riemann2Gas, "Exact Riemann Solution for 2 Perfect Gasses",
         ARG(ablate::eos::EOS, "eosL", "only valid for perfect gas"), ARG(ablate::eos::EOS, "eosR", "only valid for perfect gas"));