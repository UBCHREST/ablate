#include "riemann.hpp"
#include <eos/perfectGas.hpp>
#include "riemannCommon.hpp"


ablate::finiteVolume::fluxCalculator::Direction ablate::finiteVolume::fluxCalculator::Rieman::RiemanFluxFunction(void *ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR,
                                                                                                                 PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal *massFlux, PetscReal *p12) {
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

//PetscFPrintf(MPI_COMM_WORLD, stderr, "\x1b[1m(%s:%d, %s)\x1b[0m\n  \x1b[1m\x1b[90m\n", __FILE__, __LINE__, __FUNCTION__);
//exit(0);

    PetscInt i = 0;
    const PetscInt MAXIT = 100;
    const PetscReal err = 1e-6;
    PetscReal gamma = *(PetscReal *)ctx;  // pass-in specific heat ratio from EOS
    // This is where Rieman solver lives.
    PetscReal gamm1 = gamma - 1.0, gamp1 = gamma + 1.0;
    PetscReal pold, pstar, f_L_0, f_L_1, f_R_0, f_R_1, del_u = uR - uL;

    // Here is the initial guess for pstar - assuming two exapansion wave
    pstar = aL + aR - (0.5 * gamm1 * (uR - uL));
    pstar = pstar / ((aL / PetscPowReal(pL, 0.5 * gamm1 / gamma)) + (aR / PetscPowReal(pR, 0.5 * gamm1 / gamma)));
    pstar = PetscPowReal(pstar, 2.0 * gamma / gamm1);

    ExpansionShockCalculation(pstar, gamma, gamm1, gamp1, 0.0, pL, aL, rhoL, &f_L_0, &f_L_1);
    ExpansionShockCalculation(pstar, gamma, gamm1, gamp1, 0.0, pR, aR, rhoR, &f_R_0, &f_R_1);

    // iteration starts
    while (PetscAbsReal(f_L_0 + f_R_0 + del_u) > err && i <= MAXIT)  // Newton's method
    {
        pold = pstar;
        pstar = pold - (f_L_0 + f_R_0 + del_u) / (f_L_1 + f_R_1);  // new guess


        ExpansionShockCalculation(pstar, gamma, gamm1, gamp1, 0.0, pL, aL, rhoL, &f_L_0, &f_L_1);
        ExpansionShockCalculation(pstar, gamma, gamm1, gamp1, 0.0, pR, aR, rhoR, &f_R_0, &f_R_1);

        i++;
    }
    if (i > MAXIT) {
        throw std::runtime_error("Can't find pstar; Iteration not converging; Go back and do it again");
    }

    return riemannDirection(pstar, uL, aL, rhoL, 0.0, pL, gamma, f_L_0, uR, aR, rhoR, 0.0, pR, gamma, f_R_0, massFlux, p12);

}
ablate::finiteVolume::fluxCalculator::Rieman::Rieman(std::shared_ptr<eos::EOS> eosIn) {
    auto perfectGasEos = std::dynamic_pointer_cast<eos::PerfectGas>(eosIn);
    if (!perfectGasEos) {
        throw std::invalid_argument("ablate::flow::fluxCalculator::Direction ablate::flow::fluxCalculator::Rieman only accepts EOS of type eos::PerfectGas");
    }
    gamma = perfectGasEos->GetSpecificHeatRatio();
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::fluxCalculator::FluxCalculator, ablate::finiteVolume::fluxCalculator::Rieman, "Exact Rieman Solution", ARG(ablate::eos::EOS, "eos", "only valid for perfect gas"));
