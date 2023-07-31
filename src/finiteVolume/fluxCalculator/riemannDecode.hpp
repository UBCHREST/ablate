#ifndef ABLATELIBRARY_RIEMANNDECODE_HPP
#define ABLATELIBRARY_RIEMANNDECODE_HPP

#include <petsc.h>

void ExpansionShockCalculation(const PetscReal pstar, const PetscReal gamma, const PetscReal gamm1, const PetscReal gamp1, const PetscReal p0, const PetscReal p, const PetscReal a, const PetscReal rho, PetscReal *f0, PetscReal *f1);

void riemannDecode( const PetscReal pstar,
                    const PetscReal uL, const PetscReal aL, const PetscReal rhoL, const PetscReal p0L, const PetscReal pL, const PetscReal gammaL, const PetscReal fL,
                    const PetscReal uR, const PetscReal aR, const PetscReal rhoR, const PetscReal p0R, const PetscReal pR, const PetscReal gammaR, const PetscReal fR,
                    PetscReal *massFlux, PetscReal *p12, PetscReal *uX);

#endif  // ABLATELIBRARY_RIEMANNDECODE_HPP
