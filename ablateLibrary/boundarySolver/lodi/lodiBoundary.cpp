#include "lodiBoundary.hpp"
#include "finiteVolume/processes/flowProcess.hpp"
#include "utilities/mathUtilities.hpp"

ablate::boundarySolver::lodi::LODIBoundary::LODIBoundary(const std::shared_ptr<eos::EOS> eos) : eos(eos) {}

void ablate::boundarySolver::lodi::LODIBoundary::GetVelAndCPrims(PetscReal velNorm, PetscReal speedOfSound, PetscReal cp, PetscReal cv, PetscReal &velNormPrim, PetscReal &speedOfSoundPrim) {
    PetscReal ralpha2 = 1.;
    PetscReal fourralpha2 = 4;

    double gam = cp / cv;
    double gamm1 = gam - 1.e+0;
    double gamp1 = gam + 1.e+0;
    double M2 = velNorm / speedOfSound;
    M2 = M2 * M2;
    velNormPrim = 0.5e+0 * (velNorm * (gamp1 - gamm1));
    double gamm12 = gamm1 * gamm1;
    double tmp = 1.e+0 - ralpha2;
    tmp = tmp * tmp;
    speedOfSoundPrim = 0.5e+0 * (speedOfSound * PetscSqrtReal(gamm12 * tmp * M2 + fourralpha2));
}

void ablate::boundarySolver::lodi::LODIBoundary::GetEigenValues(PetscInt ndims, PetscInt nSpec, PetscInt nEV, PetscReal veln, PetscReal c, PetscReal velnprm, PetscReal cprm, PetscReal *lamda) {
    lamda[0] = velnprm - cprm;
    lamda[1] = veln;
    for (int ndim = 1; ndim < ndims; ndim++) {
        lamda[1 + ndim] = veln;
    }
    lamda[1 + ndims] = velnprm + cprm;
    for (int ns = 0; ns < nSpec; ns++) {
        lamda[2 + ndims + ns] = veln;
    }
    for (int ne = 0; ne < nEV; ne++) {
        lamda[2 + ndims + nSpec + ne] = veln;
    }
}

void ablate::boundarySolver::lodi::LODIBoundary::GetmdFdn(PetscInt ndims, PetscInt neqs, PetscInt nspeceq, PetscInt nEVeq, const PetscReal *vel, PetscReal rho, PetscReal T, PetscReal Cp, PetscReal Cv,
                                                          PetscReal C, PetscReal Enth, PetscReal velnprm, PetscReal Cprm, const PetscReal *Yi, const PetscReal *EV, const PetscReal *sL,
                                                          const PetscReal transformationMatrix[3][3], PetscReal *mdFdn) {
    std::vector<PetscScalar> d(neqs);
    auto fac = 0.5e+0 * (sL[0] - sL[1 + ndims]) * (velnprm - vel[0]) / Cprm;
    double C2 = C * C;
    d[0] = (sL[1] + 0.5e+0 * (sL[1 + ndims] + sL[0]) + fac) / C2;
    d[1] = 0.5e+0 * (sL[1 + ndims] + sL[0]) - fac;
    d[2] = 0.5e+0 * (sL[1 + ndims] - sL[0]) / rho / Cprm;
    for (int ndim = 1; ndim < ndims; ndim++) {
        d[2 + ndim] = sL[1 + ndim];
    }
    for (int ns = 0; ns < nspeceq; ns++) {
        d[2 + ndims + ns] = sL[2 + ndims + ns];
    }
    for (int ne = 0; ne < nEVeq; ne++) {
        d[2 + ndims + nspeceq + ne] = sL[2 + ndims + nspeceq + ne];
    }
    mdFdn[RHO] = -d[0];
    mdFdn[RHOVELN] = -(vel[0] * d[0] + rho * d[2]);  // Wall normal component momentum, not really rho u
    double KE = vel[0] * vel[0];
    double dvelterm = vel[0] * d[2];
    for (int ndim = 1; ndim < ndims; ndim++) {  // Tangential components for momentum
        mdFdn[RHOVELN + ndim] = -(vel[ndim] * d[0] + rho * d[2 + ndim]);
        KE += vel[ndim] * vel[ndim];
        dvelterm = dvelterm + vel[ndim] * d[2 + ndim];
    }
    KE = 0.5e+0 * KE;
    mdFdn[RHOE] = -(d[0] * (KE + Enth - Cp * T) + d[1] / (Cp / Cv - 1.e+0 + 1.0E-30) + rho * dvelterm);
    for (int ns = 0; ns < nspeceq; ns++) {
        mdFdn[2 + ndims + ns] = -(Yi[ns] * d[0] + rho * d[2 + ndims + ns]);  // species
    }
    for (int ne = 0; ne < nEVeq; ne++) {
        mdFdn[2 + ndims + nspeceq + ne] = -(EV[ne] * d[0] + rho * d[2 + ndims + nspeceq + ne]);  // extra
    }

    /*
        map momentum source terms (normal, tangent 1, tangent 2 back to
        physical coordinate system (x-mom, y-mom, z-mom). Note, that the
        normal direction points outward from the domain. The source term for
        the normal component of momentum is therefore for the velocity
        pointing outward from the surface. For ncomp=1 (Cartesian mesh with
        mapped space aligned with physical space) and nside=0,2 & 4 a minus
        one is multiplied by the normal component. For ncomp > 1, the dircos
        data structure is used which is more general but also more expensive.
     */
    PetscReal mdFdntmp[3] = {0.0, 0.0, 0.0};
    utilities::MathUtilities::MultiplyTranspose(ndims, transformationMatrix, mdFdn + RHOVELN, mdFdntmp);
    // Over-write source components
    for (PetscInt nc = 0; nc < ndims; nc++) {
        mdFdn[RHOVELN + nc] = mdFdntmp[nc];
    }
}
