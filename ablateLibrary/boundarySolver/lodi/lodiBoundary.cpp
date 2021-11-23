
#include "lodiBoundary.hpp"
ablate::boundarySolver::lodi::LODIBoundary::LODIBoundary(const std::shared_ptr<eos::EOS> eos) : eos(eos) {}

void ablate::boundarySolver::lodi::LODIBoundary::GetVelAndCPrims(PetscReal velNorm, PetscReal speedOfSound, PetscReal cp, PetscReal cv, PetscReal& velNormPrim, PetscReal& speedOfSoundPrim) {
    PetscReal ralpha2 = 1.;
    PetscReal fourralpha2 = 4;
    PetscReal alpha2 = 1.;

    double gam = cp / cv;
    double gamm1 = gam - 1.e+0;
    double gamp1 = gam + 1.e+0;
    double M2 = velNorm / speedOfSound;
    M2 = M2 * M2;
    velNormPrim = 0.5e+0 * (velNorm * (gamp1 - gamm1 / alpha2));
    double gamm12 = gamm1 * gamm1;
    double tmp = 1.e+0 - ralpha2;
    tmp = tmp * tmp;
    speedOfSoundPrim = 0.5e+0 * (speedOfSound * PetscSqrtReal(gamm12 * tmp * M2 + fourralpha2));
}
