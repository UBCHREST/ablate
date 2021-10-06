
#include "rieman.hpp"
#include <eos/perfectGas.hpp>
ablate::flow::fluxCalculator::Direction ablate::flow::fluxCalculator::Rieman::RiemanFluxFunction(void *ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR, PetscReal aR,
                                                                                                 PetscReal rhoR, PetscReal pR, PetscReal *massFlux,

                                                                                                 PetscReal *p12) {
    /*
     * gamma: specific heat ratio (pass in from EOS)
     * gamm1 = gamma - 1
     * gamp1 = gamma + 1
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
    PetscReal gamma = *(PetscReal *)ctx;  // pass-in specific heat ratio from EOS
    // This is where Rieman solver lives.
    PetscReal gamm1 = gamma - 1.0, gamp1 = gamma + 1.0;
    PetscReal pold, pstar, ustar, rhostarR, rhostarL, f_L_0, f_L_1, f_R_0, f_R_1, del_u = uR - uL;
    PetscReal pratio;
    PetscReal A, B, sqterm;

    // Here is the initial guess for pstar - assuming two exapansion wave
    pstar = aL + aR - (0.5 * gamm1 * (uR - uL));
    pstar = pstar / ((aL / PetscPowReal(pL, 0.5 * gamm1 / gamma)) + (aR / PetscPowReal(pR, 0.5 * gamm1 / gamma)));
    pstar = PetscPowReal(pstar, 2.0 * gamma / gamm1);

    if (pstar <= pL)  // expansion wave equation from Toto
    {
        f_L_0 = ((2. * aL) / gamm1) * (PetscPowReal(pstar / pL, 0.5 * gamm1 / gamma) - 1.);
        f_L_1 = (aL / pL / gamma) * PetscPowReal(pstar / pL, -0.5 * gamp1 / gamma);
    } else  // shock equation from Toro
    {
        A = 2 / gamp1 / rhoL;
        B = gamm1 * pL / gamp1;
        sqterm = sqrt(A / (pstar + B));
        f_L_0 = (pstar - pL) * sqterm;
        f_L_1 = sqterm * (1.0 - (0.5 * (pstar - pL) / (B + pstar)));
    }
    if (pstar <= pR)  // expansion wave equation from Toto
    {
        f_R_0 = ((2 * aR) / gamm1) * (PetscPowReal(pstar / pR, 0.5 * gamm1 / gamma) - 1);
        f_R_1 = (aR / pR / gamma) * PetscPowReal(pstar / pR, -0.5 * gamp1 / gamma);
    } else  // shock euqation from Toro
    {
        A = 2 / gamp1 / rhoR;
        B = gamm1 * pR / gamp1;
        sqterm = sqrt(A / (pstar + B));
        f_R_0 = (pstar - pR) * sqterm;
        f_R_1 = sqterm * (1.0 - (0.5 * (pstar - pR) / (B + pstar)));
    }
    // iteration starts
    while ((f_L_0 + f_R_0 + del_u) > err && i <= MAXIT)  // Newton's method
    {
        pold = pstar;
        pstar = pold - (f_L_0 + f_R_0 + del_u) / (f_L_1 + f_R_1);  // new guess
                                                                   // if ((2 * PetscAbsReal(pstar - pold) / (pstar + pold) < err)){
        //     break;
        // }  // not sure about this condition
        // else {
        if (pstar <= pL)  // expansion wave equation from Toto
        {
            f_L_0 = ((2. * aL) / gamm1) * (PetscPowReal(pstar / pL, 0.5 * gamm1 / gamma) - 1.);
            f_L_1 = (aL / pL / gamma) * PetscPowReal(pstar / pL, -0.5 * gamp1 / gamma);
        } else  // shock equation from Toro
        {
            A = 2 / gamp1 / rhoL;
            B = gamm1 * pL / gamp1;
            sqterm = sqrt(A / (pstar + B));
            f_L_0 = (pstar - pL) * sqterm;
            f_L_1 = sqterm * (1.0 - (0.5 * (pstar - pL) / (B + pstar)));
        }
        if (pstar <= pR)  // expansion wave equation from Toto
        {
            f_R_0 = ((2 * aR) / gamm1) * (PetscPowReal(pstar / pR, 0.5 * gamm1 / gamma) - 1);
            f_R_1 = (aR / pR / gamma) * PetscPowReal(pstar / pR, -0.5 * gamp1 / gamma);
        } else  // shock euqation from Toro
        {
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
    *p12 = pstar;
    // Left side
    pratio = pstar / pL;
    if (pstar <= pL)  // expansion
    {
        rhostarL = rhoL * PetscPowReal(pratio, 1.0 / gamma);
    } else  // shock
    {
        rhostarL = rhoL * (pratio + (gamm1 / gamp1)) / (gamm1 * pratio / gamp1 + 1);
    }
    // Right side
    pratio = pstar / pR;
    if (pstar <= pR)  // expansion
    {
        rhostarR = rhoR * PetscPowReal(pratio, 1 / gamma);
    } else  // shock
    {
        rhostarR = rhoR * (pratio + (gamm1 / gamp1)) / (gamm1 * pratio / gamp1 + 1);
    }
    *massFlux = 0.5 * (rhostarR + rhostarL) * ustar;
    // Check direction

    return ustar > 0 ? LEFT : RIGHT;
}
ablate::flow::fluxCalculator::Rieman::Rieman(std::shared_ptr<eos::EOS> eosIn) {
    auto perfectGasEos = std::dynamic_pointer_cast<eos::PerfectGas>(eosIn);
    if (!perfectGasEos) {
        throw std::invalid_argument("ablate::flow::fluxCalculator::Direction ablate::flow::fluxCalculator::Rieman only accepts EOS of type eos::PerfectGas");
    }
    gamma = perfectGasEos->GetSpecificHeatRatio();
}

#include "parser/registrar.hpp"
REGISTER(ablate::flow::fluxCalculator::FluxCalculator, ablate::flow::fluxCalculator::Rieman, "Exact Rieman Solution", ARG(eos::EOS, "eos", "only valid for perfect gas"));