static char help[] = "MMS from Verification of a Compressible CFD Code using the Method of Manufactured Solutions, Christopher J. Roy,† Thomas M. Smith, and Curtis C. Ober";

#include <compressibleFlow.h>
#include <petsc.h>
#include <vector>
#include "MpiTestFixture.hpp"
#include "gtest/gtest.h"

#define Pi PETSC_PI
#define Sin PetscSinReal
#define Cos PetscCosReal
#define Power PetscPowReal

typedef struct {
    PetscReal phiO;
    PetscReal phiX;
    PetscReal phiY;
    PetscReal phiZ;
    PetscReal aPhiX;
    PetscReal aPhiY;
    PetscReal aPhiZ;
} PhiConstants;

typedef struct {
    PetscInt dim;
    PhiConstants rho;
    PhiConstants u;
    PhiConstants v;
    PhiConstants w;
    PhiConstants p;
    PetscReal L;
    PetscReal gamma;
    PetscReal R;
    PetscReal mu;
} Constants;

struct CompressibleFlowMmsTestParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    Constants constants;
    PetscInt initialNx;
    std::vector<PetscReal> expectedL2Convergence;
    std::vector<PetscReal> expectedLInfConvergence;
};

typedef struct {
    Constants constants;
    FlowData flowData;
} ProblemSetup;

class CompressibleFlowMmsTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<CompressibleFlowMmsTestParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

static PetscErrorCode EulerExact(PetscInt dim, PetscReal time, const PetscReal xyz[], PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;

    Constants *constants = (Constants *)ctx;
    PetscReal L = constants->L;
    PetscReal gamma = constants->gamma;

    PetscReal rhoO = constants->rho.phiO;
    PetscReal rhoX = constants->rho.phiX;
    PetscReal rhoY = constants->rho.phiY;
    PetscReal rhoZ = constants->rho.phiZ;
    PetscReal aRhoX = constants->rho.aPhiX;
    PetscReal aRhoY = constants->rho.aPhiY;
    PetscReal aRhoZ = constants->rho.aPhiZ;

    PetscReal uO = constants->u.phiO;
    PetscReal uX = constants->u.phiX;
    PetscReal uY = constants->u.phiY;
    PetscReal uZ = constants->u.phiZ;
    PetscReal aUX = constants->u.aPhiX;
    PetscReal aUY = constants->u.aPhiY;
    PetscReal aUZ = constants->u.aPhiZ;

    PetscReal vO = constants->v.phiO;
    PetscReal vX = constants->v.phiX;
    PetscReal vY = constants->v.phiY;
    PetscReal vZ = constants->v.phiZ;
    PetscReal aVX = constants->v.aPhiX;
    PetscReal aVY = constants->v.aPhiY;
    PetscReal aVZ = constants->v.aPhiZ;

    PetscReal wO = constants->w.phiO;
    PetscReal wX = constants->w.phiX;
    PetscReal wY = constants->w.phiY;
    PetscReal wZ = constants->w.phiZ;
    PetscReal aWX = constants->w.aPhiX;
    PetscReal aWY = constants->w.aPhiY;
    PetscReal aWZ = constants->w.aPhiZ;

    PetscReal pO = constants->p.phiO;
    PetscReal pX = constants->p.phiX;
    PetscReal pY = constants->p.phiY;
    PetscReal pZ = constants->p.phiZ;
    PetscReal aPX = constants->p.aPhiX;
    PetscReal aPY = constants->p.aPhiY;
    PetscReal aPZ = constants->p.aPhiZ;

    PetscReal x = xyz[0];
    PetscReal y = xyz[1];
    PetscReal z = dim > 2 ? xyz[2] : 0.0;

    u[RHO] = rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L);
    u[RHOE] = (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) *
              ((pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L)) /
                   ((-1. + gamma) * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) +
               (Power(uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L), 2) +
                Power(wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L), 2) +
                Power(vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L), 2)) /
                   2.);

    u[RHOU + 0] = (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L)) *
                  (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L));
    u[RHOU + 1] = (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) *
                  (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L));

    if (dim > 2) {
        u[RHOU + 2] = (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L)) *
                      (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L));
    }

    PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsBoundary_Euler(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx) {
    PetscFunctionBeginUser;
    Constants *constants = (Constants *)ctx;

    // Offset the calc assuming the cells are square
    PetscReal x[3];

    for (PetscInt i = 0; i < constants->dim; i++) {
        x[i] = c[i] + n[i] * 0.5;
    }

    EulerExact(constants->dim, time, x, 0, a_xG, ctx);
    PetscFunctionReturn(0);
}

static PetscErrorCode SourceMMS(PetscInt dim, PetscReal time, const PetscReal xyz[], PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;

    Constants *constants = (Constants *)ctx;
    PetscReal L = constants->L;
    PetscReal gamma = constants->gamma;

    PetscReal rhoO = constants->rho.phiO;
    PetscReal rhoX = constants->rho.phiX;
    PetscReal rhoY = constants->rho.phiY;
    PetscReal rhoZ = constants->rho.phiZ;
    PetscReal aRhoX = constants->rho.aPhiX;
    PetscReal aRhoY = constants->rho.aPhiY;
    PetscReal aRhoZ = constants->rho.aPhiZ;

    PetscReal uO = constants->u.phiO;
    PetscReal uX = constants->u.phiX;
    PetscReal uY = constants->u.phiY;
    PetscReal uZ = constants->u.phiZ;
    PetscReal aUX = constants->u.aPhiX;
    PetscReal aUY = constants->u.aPhiY;
    PetscReal aUZ = constants->u.aPhiZ;

    PetscReal vO = constants->v.phiO;
    PetscReal vX = constants->v.phiX;
    PetscReal vY = constants->v.phiY;
    PetscReal vZ = constants->v.phiZ;
    PetscReal aVX = constants->v.aPhiX;
    PetscReal aVY = constants->v.aPhiY;
    PetscReal aVZ = constants->v.aPhiZ;

    PetscReal wO = constants->w.phiO;
    PetscReal wX = constants->w.phiX;
    PetscReal wY = constants->w.phiY;
    PetscReal wZ = constants->w.phiZ;
    PetscReal aWX = constants->w.aPhiX;
    PetscReal aWY = constants->w.aPhiY;
    PetscReal aWZ = constants->w.aPhiZ;

    PetscReal pO = constants->p.phiO;
    PetscReal pX = constants->p.phiX;
    PetscReal pY = constants->p.phiY;
    PetscReal pZ = constants->p.phiZ;
    PetscReal aPX = constants->p.aPhiX;
    PetscReal aPY = constants->p.aPhiY;
    PetscReal aPZ = constants->p.aPhiZ;

    PetscReal x = xyz[0];
    PetscReal y = xyz[1];
    PetscReal z = dim > 2 ? xyz[2] : 0.0;

    u[RHO] = (aRhoX * Pi * rhoX * Cos((aRhoX * Pi * x) / L) * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L))) / L +
             (aRhoZ * Pi * rhoZ * Cos((aRhoZ * Pi * z) / L) * (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L))) / L +
             (aUX * Pi * uX * Cos((aUX * Pi * x) / L) * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) / L +
             (aVY * Pi * vY * Cos((aVY * Pi * y) / L) * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) / L -
             (aRhoY * Pi * rhoY * Sin((aRhoY * Pi * y) / L) * (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L))) / L -
             (aWZ * Pi * wZ * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) * Sin((aWZ * Pi * z) / L)) / L;

    u[RHOE] = -((aPX * Pi * pX * Sin((aPX * Pi * x) / L) * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L))) / L) +
              (aUX * Pi * uX * Cos((aUX * Pi * x) / L) * (pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L))) / L +
              (aVY * Pi * vY * Cos((aVY * Pi * y) / L) * (pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L))) / L -
              (aPZ * Pi * pZ * (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L)) * Sin((aPZ * Pi * z) / L)) / L +
              (aPY * Pi * pY * Cos((aPY * Pi * y) / L) * (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L))) / L +
              (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) *
                  (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L)) *
                  ((aRhoY * Pi * rhoY * (pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L)) * Sin((aRhoY * Pi * y) / L)) /
                       ((-1. + gamma) * L * Power(rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L), 2)) +
                   (aPY * Pi * pY * Cos((aPY * Pi * y) / L)) / ((-1. + gamma) * L * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) +
                   ((-2 * aUY * Pi * uY * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L)) * Sin((aUY * Pi * y) / L)) / L +
                    (2 * aWY * Pi * wY * Cos((aWY * Pi * y) / L) * (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L))) / L +
                    (2 * aVY * Pi * vY * Cos((aVY * Pi * y) / L) * (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L))) / L) /
                       2.) +
              (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L)) *
                  (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) *
                  (-((aRhoX * Pi * rhoX * Cos((aRhoX * Pi * x) / L) * (pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L))) /
                     ((-1. + gamma) * L * Power(rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L), 2))) -
                   (aPX * Pi * pX * Sin((aPX * Pi * x) / L)) / ((-1. + gamma) * L * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) +
                   ((2 * aUX * Pi * uX * Cos((aUX * Pi * x) / L) * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L))) / L +
                    (2 * aWX * Pi * wX * Cos((aWX * Pi * x) / L) * (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L))) / L -
                    (2 * aVX * Pi * vX * Sin((aVX * Pi * x) / L) * (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L))) / L) /
                       2.) +
              (aRhoX * Pi * rhoX * Cos((aRhoX * Pi * x) / L) * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L)) *
               ((pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L)) /
                    ((-1. + gamma) * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) +
                (Power(uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L), 2) +
                 Power(wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L), 2) +
                 Power(vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L), 2)) /
                    2.)) /
                  L +
              (aRhoZ * Pi * rhoZ * Cos((aRhoZ * Pi * z) / L) * (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L)) *
               ((pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L)) /
                    ((-1. + gamma) * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) +
                (Power(uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L), 2) +
                 Power(wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L), 2) +
                 Power(vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L), 2)) /
                    2.)) /
                  L +
              (aUX * Pi * uX * Cos((aUX * Pi * x) / L) * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) *
               ((pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L)) /
                    ((-1. + gamma) * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) +
                (Power(uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L), 2) +
                 Power(wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L), 2) +
                 Power(vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L), 2)) /
                    2.)) /
                  L +
              (aVY * Pi * vY * Cos((aVY * Pi * y) / L) * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) *
               ((pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L)) /
                    ((-1. + gamma) * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) +
                (Power(uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L), 2) +
                 Power(wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L), 2) +
                 Power(vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L), 2)) /
                    2.)) /
                  L -
              (aRhoY * Pi * rhoY * Sin((aRhoY * Pi * y) / L) * (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L)) *
               ((pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L)) /
                    ((-1. + gamma) * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) +
                (Power(uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L), 2) +
                 Power(wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L), 2) +
                 Power(vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L), 2)) /
                    2.)) /
                  L -
              (aWZ * Pi * wZ * (pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L)) * Sin((aWZ * Pi * z) / L)) / L -
              (aWZ * Pi * wZ * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) *
               ((pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L)) /
                    ((-1. + gamma) * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) +
                (Power(uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L), 2) +
                 Power(wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L), 2) +
                 Power(vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L), 2)) /
                    2.) *
               Sin((aWZ * Pi * z) / L)) /
                  L +
              (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L)) *
                  (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) *
                  (-((aRhoZ * Pi * rhoZ * Cos((aRhoZ * Pi * z) / L) * (pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L))) /
                     ((-1. + gamma) * L * Power(rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L), 2))) -
                   (aPZ * Pi * pZ * Sin((aPZ * Pi * z) / L)) / ((-1. + gamma) * L * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) +
                   ((-2 * aUZ * Pi * uZ * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L)) * Sin((aUZ * Pi * z) / L)) / L +
                    (2 * aVZ * Pi * vZ * Cos((aVZ * Pi * z) / L) * (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L))) / L -
                    (2 * aWZ * Pi * wZ * (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L)) * Sin((aWZ * Pi * z) / L)) / L) /
                       2.);

    u[RHOU + 0] = -((aPX * Pi * pX * Sin((aPX * Pi * x) / L)) / L) +
                  (aRhoX * Pi * rhoX * Cos((aRhoX * Pi * x) / L) * Power(uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L), 2)) / L +
                  (aRhoZ * Pi * rhoZ * Cos((aRhoZ * Pi * z) / L) * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L)) *
                   (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L))) /
                      L +
                  (2 * aUX * Pi * uX * Cos((aUX * Pi * x) / L) * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L)) *
                   (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) /
                      L +
                  (aVY * Pi * vY * Cos((aVY * Pi * y) / L) * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L)) *
                   (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) /
                      L -
                  (aUZ * Pi * uZ * (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L)) *
                   (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) * Sin((aUZ * Pi * z) / L)) /
                      L -
                  (aRhoY * Pi * rhoY * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L)) * Sin((aRhoY * Pi * y) / L) *
                   (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L))) /
                      L -
                  (aUY * Pi * uY * Sin((aUY * Pi * y) / L) * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) *
                   (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L))) /
                      L -
                  (aWZ * Pi * wZ * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L)) *
                   (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) * Sin((aWZ * Pi * z) / L)) /
                      L;

    u[RHOU + 1] = (aPY * Pi * pY * Cos((aPY * Pi * y) / L)) / L -
                  (aVX * Pi * vX * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L)) * Sin((aVX * Pi * x) / L) *
                   (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) /
                      L +
                  (aVZ * Pi * vZ * Cos((aVZ * Pi * z) / L) * (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L)) *
                   (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) /
                      L +
                  (aRhoX * Pi * rhoX * Cos((aRhoX * Pi * x) / L) * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L)) *
                   (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L))) /
                      L +
                  (aRhoZ * Pi * rhoZ * Cos((aRhoZ * Pi * z) / L) * (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L)) *
                   (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L))) /
                      L +
                  (aUX * Pi * uX * Cos((aUX * Pi * x) / L) * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) *
                   (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L))) /
                      L +
                  (2 * aVY * Pi * vY * Cos((aVY * Pi * y) / L) * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) *
                   (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L))) /
                      L -
                  (aRhoY * Pi * rhoY * Sin((aRhoY * Pi * y) / L) * Power(vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L), 2)) / L -
                  (aWZ * Pi * wZ * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) *
                   (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L)) * Sin((aWZ * Pi * z) / L)) /
                      L;

    if (dim > 2) {
        u[RHOU + 2] = (aRhoX * Pi * rhoX * Cos((aRhoX * Pi * x) / L) * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L)) *
                       (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L))) /
                          L +
                      (aRhoZ * Pi * rhoZ * Cos((aRhoZ * Pi * z) / L) * Power(wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L), 2)) / L -
                      (aPZ * Pi * pZ * Sin((aPZ * Pi * z) / L)) / L +
                      (aWX * Pi * wX * Cos((aWX * Pi * x) / L) * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L)) *
                       (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) /
                          L +
                      (aUX * Pi * uX * Cos((aUX * Pi * x) / L) * (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L)) *
                       (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) /
                          L +
                      (aVY * Pi * vY * Cos((aVY * Pi * y) / L) * (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L)) *
                       (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) /
                          L -
                      (aRhoY * Pi * rhoY * Sin((aRhoY * Pi * y) / L) * (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L)) *
                       (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L))) /
                          L +
                      (aWY * Pi * wY * Cos((aWY * Pi * y) / L) * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) *
                       (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L))) /
                          L -
                      (2 * aWZ * Pi * wZ * (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L)) *
                       (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) * Sin((aWZ * Pi * z) / L)) /
                          L;
    }
    PetscFunctionReturn(0);
}

static PetscErrorCode ComputeRHSWithSourceTerms(DM dm, PetscReal time, Vec locXVec, Vec globFVec, void *ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    ProblemSetup *setup = (ProblemSetup *)ctx;

    // Call the flux calculation
    ierr = DMPlexTSComputeRHSFunctionFVM(dm, time, locXVec, globFVec, setup->flowData);
    CHKERRQ(ierr);

    // Convert the dm to a plex
    DM plex;
    DMConvert(dm, DMPLEX, &plex);

    // Extract the cell geometry, and the dm that holds the information
    Vec cellgeom;
    DM dmCell;
    const PetscScalar *cgeom;
    ierr = DMPlexGetGeometryFVM(plex, NULL, &cellgeom, NULL);
    CHKERRQ(ierr);
    ierr = VecGetDM(cellgeom, &dmCell);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellgeom, &cgeom);
    CHKERRQ(ierr);

    // Get the cell start and end for the fv cells
    PetscInt cStart, cEnd;
    ierr = DMPlexGetSimplexOrBoxCells(dmCell, 0, &cStart, &cEnd);
    CHKERRQ(ierr);

    // create a local f vector
    Vec locFVec;
    PetscScalar *locFArray;
    ierr = DMGetLocalVector(dm, &locFVec);
    CHKERRQ(ierr);
    ierr = VecZeroEntries(locFVec);
    CHKERRQ(ierr);
    ierr = VecGetArray(locFVec, &locFArray);
    CHKERRQ(ierr);

    // get the current values
    const PetscScalar *locXArray;
    ierr = VecGetArrayRead(locXVec, &locXArray);
    CHKERRQ(ierr);

    PetscInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // March over each cell volume
    for (PetscInt c = cStart; c < cEnd; ++c) {
        PetscFVCellGeom *cg;
        const PetscReal *xc;
        PetscReal *fc;

        ierr = DMPlexPointLocalRead(dmCell, c, cgeom, &cg);
        CHKERRQ(ierr);
        ierr = DMPlexPointLocalFieldRead(plex, c, 0, locXArray, &xc);
        CHKERRQ(ierr);
        ierr = DMPlexPointGlobalFieldRef(plex, c, 0, locFArray, &fc);
        CHKERRQ(ierr);

        if (fc) {  // must be real cell and not ghost
            SourceMMS(setup->constants.dim, time, cg->centroid, 0, fc + RHO, &setup->constants);
        }
    }

    // restore the cell geometry
    ierr = VecRestoreArrayRead(cellgeom, &cgeom);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(locXVec, &locXArray);
    CHKERRQ(ierr);
    ierr = VecRestoreArray(locFVec, &locFArray);
    CHKERRQ(ierr);

    ierr = DMLocalToGlobalBegin(dm, locFVec, ADD_VALUES, globFVec);
    CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm, locFVec, ADD_VALUES, globFVec);
    CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &locFVec);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode ComputeRHS(TS ts, DM dm, PetscReal t, Vec u, PetscInt blockSize, PetscReal residualNorm2[], PetscReal residualNormInf[], PetscReal start[], PetscReal end[]) {
    MPI_Comm comm;
    Vec r;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    Vec sol;
    ierr = VecDuplicate(u, &sol);
    CHKERRQ(ierr);
    ierr = VecCopy(u, sol);
    CHKERRQ(ierr);

    ierr = PetscObjectGetComm((PetscObject)ts, &comm);
    CHKERRQ(ierr);
    ierr = DMComputeExactSolution(dm, t, sol, NULL);
    CHKERRQ(ierr);
    ierr = VecDuplicate(u, &r);
    CHKERRQ(ierr);
    ierr = TSComputeRHSFunction(ts, t, sol, r);
    CHKERRQ(ierr);

    // zero out the norms
    for (PetscInt b = 0; b < blockSize; b++) {
        residualNorm2[b] = 0.0;
        residualNormInf[b] = 0.0;
    }

    {  // March over each cell
        // Extract the cell geometry, and the dm that holds the information
        Vec cellgeom;
        DM dmCell;
        PetscInt dim;
        const PetscScalar *cgeom;
        ierr = DMPlexGetGeometryFVM(dm, NULL, &cellgeom, NULL);
        CHKERRQ(ierr);
        ierr = VecGetDM(cellgeom, &dmCell);
        CHKERRQ(ierr);
        ierr = DMGetDimension(dm, &dim);
        CHKERRQ(ierr);

        // Get the cell start and end for the fv cells
        PetscInt cStart, cEnd;
        ierr = DMPlexGetSimplexOrBoxCells(dmCell, 0, &cStart, &cEnd);
        CHKERRQ(ierr);

        // temp read current residual
        const PetscScalar *currentRHS;
        ierr = VecGetArrayRead(cellgeom, &cgeom);
        CHKERRQ(ierr);
        ierr = VecGetArrayRead(r, &currentRHS);
        CHKERRQ(ierr);

        // Count up the cells
        PetscInt count = 0;

        // March over each cell volume
        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscFVCellGeom *cg;
            const PetscReal *rhsCurrent;

            ierr = DMPlexPointLocalRead(dmCell, c, cgeom, &cg);
            CHKERRQ(ierr);
            ierr = DMPlexPointGlobalFieldRead(dm, c, 0, currentRHS, &rhsCurrent);
            CHKERRQ(ierr);

            PetscBool countCell = PETSC_TRUE;
            for (PetscInt d = 0; d < dim; d++) {
                countCell = cg->centroid[d] < start[d] || cg->centroid[d] > end[d] ? PETSC_FALSE : countCell;
            }

            if (rhsCurrent && countCell) {  // must be real cell and not ghost
                for (PetscInt b = 0; b < blockSize; b++) {
                    residualNorm2[b] += PetscSqr(rhsCurrent[b]);
                    residualNormInf[b] = PetscMax(residualNormInf[b], PetscAbs(rhsCurrent[b]));
                }
                count++;
            }
        }

        // normalize the norm2
        for (PetscInt b = 0; b < blockSize; b++) {
            residualNorm2[b] = PetscSqrtReal(residualNorm2[b] / count);
        }

        // temp return current residual
        ierr = VecRestoreArrayRead(cellgeom, &cgeom);
        CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(r, &currentRHS);
        CHKERRQ(ierr);
    }

    ierr = VecDestroy(&sol);
    CHKERRQ(ierr);
    ierr = VecDestroy(&r);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

TEST_P(CompressibleFlowMmsTestFixture, ShouldComputeCorrectFlux) {
    StartWithMPI
        PetscErrorCode ierr;

        // initialize petsc and mpi
        PetscInitialize(argc, argv, NULL, "HELP") >> errorChecker;

        PetscInt levels = 4;

        Constants constants = GetParam().constants;
        PetscInt blockSize = 2 + constants.dim;
        PetscInt initialNx = GetParam().initialNx;

        std::vector<PetscReal> hHistory;
        std::vector<std::vector<PetscReal>> l2History(blockSize);
        std::vector<std::vector<PetscReal>> lInfHistory(blockSize);

        // March over each level
        for (PetscInt l = 0; l < levels; l++) {
            PetscPrintf(PETSC_COMM_WORLD, "Running RHS Calculation at Level %d", l);

            DM dm; /* problem definition */
            TS ts; /* timestepper */

            // Create a ts
            TSCreate(PETSC_COMM_WORLD, &ts) >> errorChecker;
            TSSetProblemType(ts, TS_NONLINEAR) >> errorChecker;
            TSSetType(ts, TSEULER) >> errorChecker;
            TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP) >> errorChecker;

            // Create a mesh
            // hard code the problem setup
            PetscReal start[] = {0.0, 0.0, 0.0};
            PetscReal end[] = {constants.L, constants.L, constants.L};
            PetscInt nx1D = initialNx * PetscPowRealInt(2, l);
            PetscInt nx[] = {nx1D, nx1D, nx1D};
            DMBoundaryType bcType[] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
            DMPlexCreateBoxMesh(PETSC_COMM_WORLD, constants.dim, PETSC_FALSE, nx, start, end, bcType, PETSC_TRUE, &dm) >> errorChecker;

            // Setup the flow data
            FlowData flowData; /* store some of the flow data*/
            FlowCreate(&flowData) >> errorChecker;

            // Combine the flow data
            ProblemSetup problemSetup;
            problemSetup.flowData = flowData;
            problemSetup.constants = constants;

            // Setup
            CompressibleFlow_SetupDiscretization(flowData, &dm);

            // Add in the flow parameters
            PetscScalar params[TOTAL_COMPRESSIBLE_FLOW_PARAMETERS];
            params[CFL] = 0.5;
            params[GAMMA] = constants.gamma;

            // set up the finite volume fluxes
            CompressibleFlow_StartProblemSetup(flowData, TOTAL_COMPRESSIBLE_FLOW_PARAMETERS, params) >> errorChecker;

            // Add in any boundary conditions
            PetscDS prob;
            ierr = DMGetDS(flowData->dm, &prob);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            const PetscInt idsLeft[] = {1, 2, 3, 4};
            PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "wall left", "Face Sets", 0, 0, NULL, (void (*)(void))PhysicsBoundary_Euler, NULL, 4, idsLeft, &constants) >> errorChecker;

            // Complete the problem setup
            CompressibleFlow_CompleteProblemSetup(flowData, ts) >> errorChecker;

            // Override the flow calc for now
            DMTSSetRHSFunctionLocal(flowData->dm, ComputeRHSWithSourceTerms, &problemSetup) >> errorChecker;

            // Name the flow field
            PetscObjectSetName(((PetscObject)flowData->flowField), "Numerical Solution") >> errorChecker;

            // Setup the TS
            TSSetFromOptions(ts) >> errorChecker;

            // set the initial conditions
            PetscErrorCode (*func[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) = {EulerExact};
            void *ctxs[1] = {&constants};
            DMProjectFunction(flowData->dm, 0.0, func, ctxs, INSERT_ALL_VALUES, flowData->flowField) >> errorChecker;

            // for the mms, add the exact solution
            PetscDSSetExactSolution(prob, 0, EulerExact, &constants) >> errorChecker;

            TSSetMaxSteps(ts, 1);
            TSSolve(ts, flowData->flowField) >> errorChecker;

            // Check the current residual
            PetscReal l2Residual[5];
            PetscReal infResidual[5];

            // Only take the residual over the central 1/3
            PetscReal resStart[3] = {constants.L / 3.0, constants.L / 3.0, constants.L / 3.0};
            PetscReal resEnd[3] = {2.0 * constants.L / 3.0, 2.0 * constants.L / 3.0, 2.0 * constants.L / 3.0};

            ComputeRHS(ts, flowData->dm, 0.0, flowData->flowField, blockSize, l2Residual, infResidual, resStart, resEnd) >> errorChecker;
            PetscPrintf(PETSC_COMM_WORLD, "\tL_2 Residual: [%2.3g, %2.3g, %2.3g, %2.3g]\n", (double)l2Residual[0], (double)l2Residual[1], (double)l2Residual[2], (double)l2Residual[3]) >> errorChecker;
            PetscPrintf(PETSC_COMM_WORLD, "\tL_Inf Residual: [%2.3g, %2.3g, %2.3g, %2.3g]\n", (double)infResidual[0], (double)infResidual[1], (double)infResidual[2], (double)infResidual[3]) >>
                errorChecker;

            // Store the residual into history
            hHistory.push_back(PetscLog10Real(constants.L / nx1D));
            for (auto b = 0; b < blockSize; b++) {
                l2History[b].push_back(PetscLog10Real(l2Residual[b]));
                lInfHistory[b].push_back(PetscLog10Real(infResidual[b]));
            }

            FlowDestroy(&flowData) >> errorChecker;
            TSDestroy(&ts) >> errorChecker;
        }

        // Fit each component and output
        for (auto b = 0; b < blockSize; b++) {
            PetscReal l2Slope;
            PetscReal l2Intercept;
            PetscLinearRegression(hHistory.size(), &hHistory[0], &l2History[b][0], &l2Slope, &l2Intercept) >> errorChecker;

            PetscReal lInfSlope;
            PetscReal lInfIntercept;
            PetscLinearRegression(hHistory.size(), &hHistory[0], &lInfHistory[b][0], &lInfSlope, &lInfIntercept) >> errorChecker;

            PetscPrintf(PETSC_COMM_WORLD, "RHS Convergence[%d]: L2 %2.3g LInf %2.3g \n", b, l2Slope, lInfSlope) >> errorChecker;

            ASSERT_NEAR(l2Slope, GetParam().expectedL2Convergence[b], 0.2) << "incorrect L2 convergence order for component[" << b << "]";
            ASSERT_NEAR(lInfSlope, GetParam().expectedLInfConvergence[b], 0.2) << "incorrect LInf convergence order for component[" << b << "]";
        }

        ierr = PetscFinalize();
        exit(ierr);

    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    CompressibleFlow, CompressibleFlowMmsTestFixture,
    testing::Values((CompressibleFlowMmsTestParameters){.mpiTestParameter = {.testName = "low speed average", .nproc = 1, .arguments = "-dm_plex_separate_marker -flux_diff average"},
                                                        .constants = {.dim = 2,
                                                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                                                      .u = {.phiO = 70, .phiX = 5, .phiY = -7, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                                                      .v = {.phiO = 90, .phiX = -15, .phiY = -8.5, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                                                      .L = 1.0,
                                                                      .gamma = 1.4},
                                                        .initialNx = 4,
                                                        .expectedL2Convergence = {2, 2, 2, 2},
                                                        .expectedLInfConvergence = {1.9, 1.8, 1.8, 1.8}},
                    (CompressibleFlowMmsTestParameters){.mpiTestParameter = {.testName = "high speed average", .nproc = 1, .arguments = "-dm_plex_separate_marker -flux_diff average"},
                                                        .constants = {.dim = 2,
                                                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                                                      .u = {.phiO = 800, .phiX = 50, .phiY = -30.0, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                                                      .v = {.phiO = 800, .phiX = -75, .phiY = 40, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                                                      .L = 1.0,
                                                                      .gamma = 1.4},
                                                        .initialNx = 4,
                                                        .expectedL2Convergence = {2, 2, 2, 2},
                                                        .expectedLInfConvergence = {1.9, 1.8, 1.8, 1.8}},
                    (CompressibleFlowMmsTestParameters){.mpiTestParameter = {.testName = "low speed ausm", .nproc = 1, .arguments = "-dm_plex_separate_marker -flux_diff ausm"},
                                                        .constants = {.dim = 2,
                                                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                                                      .u = {.phiO = 70, .phiX = 5, .phiY = -7, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                                                      .v = {.phiO = 90, .phiX = -15, .phiY = -8.5, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                                                      .L = 1.0,
                                                                      .gamma = 1.4},
                                                        .initialNx = 16,
                                                        .expectedL2Convergence = {1.0, 1.0, 1.4, 1.0},
                                                        .expectedLInfConvergence = {1.0, 1.0, 1.4, 1.0}},
                    (CompressibleFlowMmsTestParameters){.mpiTestParameter = {.testName = "high speed ausm", .nproc = 1, .arguments = "-dm_plex_separate_marker -flux_diff ausm"},
                                                        .constants = {.dim = 2,
                                                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                                                      .u = {.phiO = 800, .phiX = 50, .phiY = -30.0, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                                                      .v = {.phiO = 800, .phiX = -75, .phiY = 40, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                                                      .L = 1.0,
                                                                      .gamma = 1.4},
                                                        .initialNx = 16,
                                                        .expectedL2Convergence = {1.0, 1.0, 1.0, 1.0},
                                                        .expectedLInfConvergence = {1.0, 1.0, 1.0, 1.0}},
                    (CompressibleFlowMmsTestParameters){
                        .mpiTestParameter = {.testName = "low speed ausm leastsquares", .nproc = 1, .arguments = "-dm_plex_separate_marker -flux_diff ausm -eulerpetscfv_type leastsquares"},
                        .constants = {.dim = 2,
                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                      .u = {.phiO = 70, .phiX = 5, .phiY = -7, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                      .v = {.phiO = 90, .phiX = -15, .phiY = -8.5, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                      .L = 1.0,
                                      .gamma = 1.4},
                        .initialNx = 16,
                        .expectedL2Convergence = {1.5, 1.5, 1.5, 1.5},
                        .expectedLInfConvergence = {1.0, 1.0, 1.0, 1.0}},
                    (CompressibleFlowMmsTestParameters){
                        .mpiTestParameter = {.testName = "high speed ausm leastsquares", .nproc = 1, .arguments = "-dm_plex_separate_marker -flux_diff ausm -eulerpetscfv_type leastsquares"},
                        .constants = {.dim = 2,
                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                      .u = {.phiO = 800, .phiX = 50, .phiY = -30.0, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                      .v = {.phiO = 800, .phiX = -75, .phiY = 40, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                      .L = 1.0,
                                      .gamma = 1.4},
                        .initialNx = 16,
                        .expectedL2Convergence = {1.5, 1.5, 1.5, 1.5},
                        .expectedLInfConvergence = {1.0, 0.5, 1.0, 1.0}}),
    [](const testing::TestParamInfo<CompressibleFlowMmsTestParameters> &info) { return info.param.mpiTestParameter.getTestName(); });
