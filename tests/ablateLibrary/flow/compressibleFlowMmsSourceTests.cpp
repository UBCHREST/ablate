#include <petsc.h>
#include <flow/processes/eulerAdvection.hpp>
#include <mesh/dmWrapper.hpp>
#include <vector>
#include "MpiTestFixture.hpp"
#include "eos/perfectGas.hpp"
#include "flow/boundaryConditions/ghost.hpp"
#include "flow/compressibleFlow.hpp"
#include "flow/fluxCalculator/ausm.hpp"
#include "flow/fluxCalculator/ausmpUp.hpp"
#include "flow/fluxCalculator/averageFlux.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "parameters/mapParameters.hpp"

#define Pi PETSC_PI
#define Sin PetscSinReal
#define Cos PetscCosReal
#define Power PetscPowReal

using namespace ablate;

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
    PetscReal k;
} Constants;

struct CompressibleFlowMmsTestParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    std::shared_ptr<flow::fluxCalculator::FluxCalculator> fluxCalculator;
    Constants constants;
    PetscInt initialNx;
    PetscInt levels;
    std::vector<PetscReal> expectedL2Convergence;
    std::vector<PetscReal> expectedLInfConvergence;
};

typedef struct {
    Constants constants;
    std::shared_ptr<ablate::flow::CompressibleFlow> flowData;
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

    u[ablate::flow::processes::EulerAdvection::RHO] = rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L);
    u[ablate::flow::processes::EulerAdvection::RHOE] = (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) *
                                                       ((pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L)) /
                                                            ((-1. + gamma) * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) +
                                                        (Power(uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L), 2) +
                                                         Power(wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L), 2) +
                                                         Power(vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L), 2)) /
                                                            2.);

    u[ablate::flow::processes::EulerAdvection::RHOU + 0] = (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L)) *
                                                           (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L));
    u[ablate::flow::processes::EulerAdvection::RHOU + 1] = (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) *
                                                           (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L));

    if (dim > 2) {
        u[ablate::flow::processes::EulerAdvection::RHOU + 2] = (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L)) *
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

static PetscErrorCode SourceMMS(PetscInt dim, const PetscFVCellGeom *cg, const PetscInt uOff[], const PetscScalar u[], const PetscInt aOff[], const PetscScalar a[], PetscScalar f[], void *ctx) {
    PetscFunctionBeginUser;

    Constants *constants = (Constants *)ctx;
    PetscReal L = constants->L;
    PetscReal gamma = constants->gamma;
    PetscReal R = constants->R;
    PetscReal k = constants->k;
    PetscReal mu = constants->mu;

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

    PetscReal x = cg->centroid[0];
    PetscReal y = cg->centroid[1];
    PetscReal z = dim > 2 ? cg->centroid[2] : 0.0;

    f[ablate::flow::processes::EulerAdvection::RHO] =
        (aRhoX * Pi * rhoX * Cos((aRhoX * Pi * x) / L) * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L))) / L +
        (aRhoZ * Pi * rhoZ * Cos((aRhoZ * Pi * z) / L) * (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L))) / L +
        (aUX * Pi * uX * Cos((aUX * Pi * x) / L) * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) / L +
        (aVY * Pi * vY * Cos((aVY * Pi * y) / L) * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) / L -
        (aRhoY * Pi * rhoY * Sin((aRhoY * Pi * y) / L) * (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L))) / L -
        (aWZ * Pi * wZ * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)) * Sin((aWZ * Pi * z) / L)) / L;

    f[ablate::flow::processes::EulerAdvection::RHOE] =
        -((aWY * mu * Pi * wY * Cos((aWY * Pi * y) / L) * ((aWY * Pi * wY * Cos((aWY * Pi * y) / L)) / L + (aVZ * Pi * vZ * Cos((aVZ * Pi * z) / L)) / L)) / L) -
        (aVZ * mu * Pi * vZ * Cos((aVZ * Pi * z) / L) * ((aWY * Pi * wY * Cos((aWY * Pi * y) / L)) / L + (aVZ * Pi * vZ * Cos((aVZ * Pi * z) / L)) / L)) / L +
        (Power(aUY, 2) * mu * Power(Pi, 2) * uY * Cos((aUY * Pi * y) / L) * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L))) / Power(L, 2) +
        (Power(aUZ, 2) * mu * Power(Pi, 2) * uZ * Cos((aUZ * Pi * z) / L) * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L))) / Power(L, 2) -
        (aPX * Pi * pX * Sin((aPX * Pi * x) / L) * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L))) / L +
        (4 * Power(aUX, 2) * mu * Power(Pi, 2) * uX * Sin((aUX * Pi * x) / L) * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L))) /
            (3. * Power(L, 2)) +
        (aUX * Pi * uX * Cos((aUX * Pi * x) / L) * (pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L))) / L +
        (aVY * Pi * vY * Cos((aVY * Pi * y) / L) * (pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L))) / L +
        (aVX * mu * Pi * vX * Sin((aVX * Pi * x) / L) * (-((aVX * Pi * vX * Sin((aVX * Pi * x) / L)) / L) - (aUY * Pi * uY * Sin((aUY * Pi * y) / L)) / L)) / L +
        (aUY * mu * Pi * uY * Sin((aUY * Pi * y) / L) * (-((aVX * Pi * vX * Sin((aVX * Pi * x) / L)) / L) - (aUY * Pi * uY * Sin((aUY * Pi * y) / L)) / L)) / L +
        (4 * Power(aWZ, 2) * mu * Power(Pi, 2) * wZ * Cos((aWZ * Pi * z) / L) * (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L))) /
            (3. * Power(L, 2)) +
        (Power(aWX, 2) * mu * Power(Pi, 2) * wX * Sin((aWX * Pi * x) / L) * (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L))) / Power(L, 2) +
        (Power(aWY, 2) * mu * Power(Pi, 2) * wY * Sin((aWY * Pi * y) / L) * (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L))) / Power(L, 2) -
        (aPZ * Pi * pZ * (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L)) * Sin((aPZ * Pi * z) / L)) / L -
        k * ((2 * Power(aRhoX, 2) * Power(Pi, 2) * Power(rhoX, 2) * Power(Cos((aRhoX * Pi * x) / L), 2) *
              (pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L))) /
                 (Power(L, 2) * R * Power(rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L), 3)) +
             (2 * aPX * aRhoX * Power(Pi, 2) * pX * rhoX * Cos((aRhoX * Pi * x) / L) * Sin((aPX * Pi * x) / L)) /
                 (Power(L, 2) * R * Power(rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L), 2)) +
             (Power(aRhoX, 2) * Power(Pi, 2) * rhoX * Sin((aRhoX * Pi * x) / L) * (pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L))) /
                 (Power(L, 2) * R * Power(rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L), 2)) -
             (Power(aPX, 2) * Power(Pi, 2) * pX * Cos((aPX * Pi * x) / L)) /
                 (Power(L, 2) * R * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)))) -
        k * ((2 * Power(aRhoZ, 2) * Power(Pi, 2) * Power(rhoZ, 2) * Power(Cos((aRhoZ * Pi * z) / L), 2) *
              (pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L))) /
                 (Power(L, 2) * R * Power(rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L), 3)) +
             (2 * aPZ * aRhoZ * Power(Pi, 2) * pZ * rhoZ * Cos((aRhoZ * Pi * z) / L) * Sin((aPZ * Pi * z) / L)) /
                 (Power(L, 2) * R * Power(rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L), 2)) +
             (Power(aRhoZ, 2) * Power(Pi, 2) * rhoZ * (pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L)) * Sin((aRhoZ * Pi * z) / L)) /
                 (Power(L, 2) * R * Power(rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L), 2)) -
             (Power(aPZ, 2) * Power(Pi, 2) * pZ * Cos((aPZ * Pi * z) / L)) /
                 (Power(L, 2) * R * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)))) -
        k * ((2 * Power(aRhoY, 2) * Power(Pi, 2) * Power(rhoY, 2) * (pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L)) *
              Power(Sin((aRhoY * Pi * y) / L), 2)) /
                 (Power(L, 2) * R * Power(rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L), 3)) +
             (Power(aRhoY, 2) * Power(Pi, 2) * rhoY * Cos((aRhoY * Pi * y) / L) * (pO + pX * Cos((aPX * Pi * x) / L) + pZ * Cos((aPZ * Pi * z) / L) + pY * Sin((aPY * Pi * y) / L))) /
                 (Power(L, 2) * R * Power(rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L), 2)) +
             (2 * aPY * aRhoY * Power(Pi, 2) * pY * rhoY * Cos((aPY * Pi * y) / L) * Sin((aRhoY * Pi * y) / L)) /
                 (Power(L, 2) * R * Power(rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L), 2)) -
             (Power(aPY, 2) * Power(Pi, 2) * pY * Sin((aPY * Pi * y) / L)) /
                 (Power(L, 2) * R * (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L)))) -
        (aWX * mu * Pi * wX * Cos((aWX * Pi * x) / L) * ((aWX * Pi * wX * Cos((aWX * Pi * x) / L)) / L - (aUZ * Pi * uZ * Sin((aUZ * Pi * z) / L)) / L)) / L +
        (aUZ * mu * Pi * uZ * Sin((aUZ * Pi * z) / L) * ((aWX * Pi * wX * Cos((aWX * Pi * x) / L)) / L - (aUZ * Pi * uZ * Sin((aUZ * Pi * z) / L)) / L)) / L +
        (Power(aVX, 2) * mu * Power(Pi, 2) * vX * Cos((aVX * Pi * x) / L) * (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L))) / Power(L, 2) +
        (aPY * Pi * pY * Cos((aPY * Pi * y) / L) * (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L))) / L +
        (4 * Power(aVY, 2) * mu * Power(Pi, 2) * vY * Sin((aVY * Pi * y) / L) * (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L))) /
            (3. * Power(L, 2)) +
        (Power(aVZ, 2) * mu * Power(Pi, 2) * vZ * Sin((aVZ * Pi * z) / L) * (vO + vX * Cos((aVX * Pi * x) / L) + vY * Sin((aVY * Pi * y) / L) + vZ * Sin((aVZ * Pi * z) / L))) / Power(L, 2) +
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
            L -
        (2 * aUX * mu * Pi * uX * Cos((aUX * Pi * x) / L) *
         ((aUX * Pi * uX * Cos((aUX * Pi * x) / L)) / L +
          (-((aUX * Pi * uX * Cos((aUX * Pi * x) / L)) / L) - (aVY * Pi * vY * Cos((aVY * Pi * y) / L)) / L + (aWZ * Pi * wZ * Sin((aWZ * Pi * z) / L)) / L) / 3.)) /
            L -
        (2 * aVY * mu * Pi * vY * Cos((aVY * Pi * y) / L) *
         ((aVY * Pi * vY * Cos((aVY * Pi * y) / L)) / L +
          (-((aUX * Pi * uX * Cos((aUX * Pi * x) / L)) / L) - (aVY * Pi * vY * Cos((aVY * Pi * y) / L)) / L + (aWZ * Pi * wZ * Sin((aWZ * Pi * z) / L)) / L) / 3.)) /
            L +
        (2 * aWZ * mu * Pi * wZ * Sin((aWZ * Pi * z) / L) *
         (-((aWZ * Pi * wZ * Sin((aWZ * Pi * z) / L)) / L) +
          (-((aUX * Pi * uX * Cos((aUX * Pi * x) / L)) / L) - (aVY * Pi * vY * Cos((aVY * Pi * y) / L)) / L + (aWZ * Pi * wZ * Sin((aWZ * Pi * z) / L)) / L) / 3.)) /
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

    f[ablate::flow::processes::EulerAdvection::RHOU + 0] =
        (Power(aUY, 2) * mu * Power(Pi, 2) * uY * Cos((aUY * Pi * y) / L)) / Power(L, 2) + (Power(aUZ, 2) * mu * Power(Pi, 2) * uZ * Cos((aUZ * Pi * z) / L)) / Power(L, 2) -
        (aPX * Pi * pX * Sin((aPX * Pi * x) / L)) / L + (4 * Power(aUX, 2) * mu * Power(Pi, 2) * uX * Sin((aUX * Pi * x) / L)) / (3. * Power(L, 2)) +
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

    f[ablate::flow::processes::EulerAdvection::RHOU + 1] =
        (Power(aVX, 2) * mu * Power(Pi, 2) * vX * Cos((aVX * Pi * x) / L)) / Power(L, 2) + (aPY * Pi * pY * Cos((aPY * Pi * y) / L)) / L +
        (4 * Power(aVY, 2) * mu * Power(Pi, 2) * vY * Sin((aVY * Pi * y) / L)) / (3. * Power(L, 2)) -
        (aVX * Pi * vX * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L)) * Sin((aVX * Pi * x) / L) *
         (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) /
            L +
        (aVZ * Pi * vZ * Cos((aVZ * Pi * z) / L) * (wO + wZ * Cos((aWZ * Pi * z) / L) + wX * Sin((aWX * Pi * x) / L) + wY * Sin((aWY * Pi * y) / L)) *
         (rhoO + rhoY * Cos((aRhoY * Pi * y) / L) + rhoX * Sin((aRhoX * Pi * x) / L) + rhoZ * Sin((aRhoZ * Pi * z) / L))) /
            L +
        (Power(aVZ, 2) * mu * Power(Pi, 2) * vZ * Sin((aVZ * Pi * z) / L)) / Power(L, 2) +
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
        f[ablate::flow::processes::EulerAdvection::RHOU + 2] =
            (4 * Power(aWZ, 2) * mu * Power(Pi, 2) * wZ * Cos((aWZ * Pi * z) / L)) / (3. * Power(L, 2)) + (Power(aWX, 2) * mu * Power(Pi, 2) * wX * Sin((aWX * Pi * x) / L)) / Power(L, 2) +
            (Power(aWY, 2) * mu * Power(Pi, 2) * wY * Sin((aWY * Pi * y) / L)) / Power(L, 2) +
            (aRhoX * Pi * rhoX * Cos((aRhoX * Pi * x) / L) * (uO + uY * Cos((aUY * Pi * y) / L) + uZ * Cos((aUZ * Pi * z) / L) + uX * Sin((aUX * Pi * x) / L)) *
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
        PetscInitialize(argc, argv, NULL, "HELP") >> testErrorChecker;

        Constants constants = GetParam().constants;
        PetscInt blockSize = 2 + constants.dim;
        PetscInt initialNx = GetParam().initialNx;
        PetscInt levels = GetParam().levels;

        std::vector<PetscReal> hHistory;
        std::vector<std::vector<PetscReal>> l2History(blockSize);
        std::vector<std::vector<PetscReal>> lInfHistory(blockSize);

        // March over each level
        for (PetscInt l = 0; l < levels; l++) {
            PetscPrintf(PETSC_COMM_WORLD, "Running RHS Calculation at Level %d\n", l);

            DM dmCreate; /* problem definition */
            TS ts;       /* timestepper */

            // Create a ts
            TSCreate(PETSC_COMM_WORLD, &ts) >> testErrorChecker;
            TSSetProblemType(ts, TS_NONLINEAR) >> testErrorChecker;
            TSSetType(ts, TSEULER) >> testErrorChecker;
            TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP) >> testErrorChecker;

            // Create a mesh
            // hard code the problem setup
            PetscReal start[] = {0.0, 0.0, 0.0};
            PetscReal end[] = {constants.L, constants.L, constants.L};
            PetscInt nx1D = initialNx * PetscPowRealInt(2, l);
            PetscInt nx[] = {nx1D, nx1D, nx1D};
            DMBoundaryType bcType[] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
            DMPlexCreateBoxMesh(PETSC_COMM_WORLD, constants.dim, PETSC_FALSE, nx, start, end, bcType, PETSC_TRUE, &dmCreate) >> testErrorChecker;

            // Setup the flow data
            auto parameters =
                std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"cfl", "0.5"}, {"mu", std::to_string(constants.mu)}, {"k", std::to_string(constants.k)}});

            auto eos = std::make_shared<ablate::eos::PerfectGas>(
                std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", std::to_string(constants.gamma)}, {"Rgas", std::to_string(constants.R)}}));

            auto exactSolution = std::make_shared<mathFunctions::FieldSolution>("euler", mathFunctions::Create(EulerExact, &constants));

            auto boundaryConditions = std::vector<std::shared_ptr<flow::boundaryConditions::BoundaryCondition>>{
                std::make_shared<flow::boundaryConditions::Ghost>("euler", "walls", "Face Sets", std::vector<int>{1, 2, 3, 4}, PhysicsBoundary_Euler, &constants),
            };

            auto flowObject = std::make_shared<ablate::flow::CompressibleFlow>("testFlow",
                                                                               std::make_shared<ablate::mesh::DMWrapper>(dmCreate),
                                                                               eos,
                                                                               parameters,
                                                                               GetParam().fluxCalculator,
                                                                               nullptr /*options*/,
                                                                               std::vector<std::shared_ptr<mathFunctions::FieldSolution>>{exactSolution} /*initialization*/,
                                                                               boundaryConditions /*boundary conditions*/,
                                                                               std::vector<std::shared_ptr<mathFunctions::FieldSolution>>{exactSolution} /*exactSolution*/);

            // Combine the flow data
            ProblemSetup problemSetup;
            problemSetup.flowData = flowObject;
            problemSetup.constants = constants;

            // Complete the problem setup
            flowObject->CompleteProblemSetup(ts);

            // Add a point wise function that adds fluxes to euler.  It requires no input fields
            flowObject->RegisterRHSFunction(SourceMMS, &problemSetup, {"euler"}, {}, {});

            // Name the flow field
            PetscObjectSetName(((PetscObject)flowObject->GetSolutionVector()), "Numerical Solution") >> testErrorChecker;

            // Setup the TS
            TSSetFromOptions(ts) >> testErrorChecker;
            TSSetMaxSteps(ts, 1);
            TSSolve(ts, flowObject->GetSolutionVector()) >> testErrorChecker;

            // Check the current residual
            std::vector<PetscReal> l2Residual(blockSize);
            std::vector<PetscReal> infResidual(blockSize);

            // Only take the residual over the central 1/3
            PetscReal resStart[3] = {constants.L / 3.0, constants.L / 3.0, constants.L / 3.0};
            PetscReal resEnd[3] = {2.0 * constants.L / 3.0, 2.0 * constants.L / 3.0, 2.0 * constants.L / 3.0};

            ComputeRHS(ts, flowObject->GetDM(), 0.0, flowObject->GetSolutionVector(), blockSize, &l2Residual[0], &infResidual[0], resStart, resEnd) >> testErrorChecker;
            auto l2String = PrintVector(l2Residual, "%2.3g");
            PetscPrintf(PETSC_COMM_WORLD, "\tL_2 Residual: %s\n", l2String.c_str()) >> testErrorChecker;
            auto lInfString = PrintVector(infResidual, "%2.3g");
            PetscPrintf(PETSC_COMM_WORLD, "\tL_Inf Residual: %s\n", lInfString.c_str()) >> testErrorChecker;

            // Store the residual into history
            hHistory.push_back(PetscLog10Real(constants.L / nx1D));
            for (auto b = 0; b < blockSize; b++) {
                l2History[b].push_back(PetscLog10Real(l2Residual[b]));
                lInfHistory[b].push_back(PetscLog10Real(infResidual[b]));
            }
            TSDestroy(&ts) >> testErrorChecker;
        }

        // Fit each component and output
        for (auto b = 0; b < blockSize; b++) {
            PetscReal l2Slope;
            PetscReal l2Intercept;
            PetscLinearRegression(hHistory.size(), &hHistory[0], &l2History[b][0], &l2Slope, &l2Intercept) >> testErrorChecker;

            PetscReal lInfSlope;
            PetscReal lInfIntercept;
            PetscLinearRegression(hHistory.size(), &hHistory[0], &lInfHistory[b][0], &lInfSlope, &lInfIntercept) >> testErrorChecker;

            PetscPrintf(PETSC_COMM_WORLD, "RHS Convergence[%d]: L2 %2.3g LInf %2.3g \n", b, l2Slope, lInfSlope) >> testErrorChecker;

            ASSERT_NEAR(l2Slope, GetParam().expectedL2Convergence[b], 0.2) << "incorrect L2 convergence order for component[" << b << "]";
            ASSERT_NEAR(lInfSlope, GetParam().expectedLInfConvergence[b], 0.2) << "incorrect LInf convergence order for component[" << b << "]";
        }

        ierr = PetscFinalize();
        exit(ierr);

    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    CompressibleFlow, CompressibleFlowMmsTestFixture,
    testing::Values((CompressibleFlowMmsTestParameters){.mpiTestParameter = {.testName = "low speed average", .nproc = 1, .arguments = "-dm_plex_separate_marker"},
                                                        .fluxCalculator = std::make_shared<ablate::flow::fluxCalculator::AverageFlux>(),
                                                        .constants = {.dim = 2,
                                                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                                                      .u = {.phiO = 70, .phiX = 5, .phiY = -7, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                                                      .v = {.phiO = 90, .phiX = -15, .phiY = -8.5, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                                                      .L = 1.0,
                                                                      .gamma = 1.4,
                                                                      .R = 287.0,
                                                                      .mu = 0.0,
                                                                      .k = 0.0},
                                                        .initialNx = 4,
                                                        .levels = 4,
                                                        .expectedL2Convergence = {2, 2, 2, 2},
                                                        .expectedLInfConvergence = {1.9, 1.8, 1.8, 1.8}},
                    (CompressibleFlowMmsTestParameters){.mpiTestParameter = {.testName = "high speed average", .nproc = 1, .arguments = "-dm_plex_separate_marker"},
                                                        .fluxCalculator = std::make_shared<ablate::flow::fluxCalculator::AverageFlux>(),

                                                        .constants = {.dim = 2,
                                                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                                                      .u = {.phiO = 800, .phiX = 50, .phiY = -30.0, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                                                      .v = {.phiO = 800, .phiX = -75, .phiY = 40, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                                                      .L = 1.0,
                                                                      .gamma = 1.4,
                                                                      .R = 287.0,
                                                                      .mu = 0.0,
                                                                      .k = 0.0},
                                                        .initialNx = 4,
                                                        .levels = 4,
                                                        .expectedL2Convergence = {2, 2, 2, 2},
                                                        .expectedLInfConvergence = {1.9, 1.8, 1.8, 1.8}},
                    (CompressibleFlowMmsTestParameters){.mpiTestParameter = {.testName = "low speed ausm", .nproc = 1, .arguments = "-dm_plex_separate_marker"},
                                                        .fluxCalculator = std::make_shared<ablate::flow::fluxCalculator::Ausm>(),

                                                        .constants = {.dim = 2,
                                                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                                                      .u = {.phiO = 70, .phiX = 5, .phiY = -7, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                                                      .v = {.phiO = 90, .phiX = -15, .phiY = -8.5, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                                                      .L = 1.0,
                                                                      .gamma = 1.4,
                                                                      .R = 287.0,
                                                                      .mu = 0.0,
                                                                      .k = 0.0},
                                                        .initialNx = 16,
                                                        .levels = 4,
                                                        .expectedL2Convergence = {1.0, 1.0, 1.4, 1.0},
                                                        .expectedLInfConvergence = {1.0, 1.0, 1.4, 1.0}},
                    (CompressibleFlowMmsTestParameters){.mpiTestParameter = {.testName = "high speed ausm", .nproc = 1, .arguments = "-dm_plex_separate_marker "},
                                                        .fluxCalculator = std::make_shared<ablate::flow::fluxCalculator::Ausm>(),

                                                        .constants = {.dim = 2,
                                                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                                                      .u = {.phiO = 800, .phiX = 50, .phiY = -30.0, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                                                      .v = {.phiO = 800, .phiX = -75, .phiY = 40, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                                                      .L = 1.0,
                                                                      .gamma = 1.4,
                                                                      .R = 287.0,
                                                                      .mu = 0.0,
                                                                      .k = 0.0},
                                                        .initialNx = 16,
                                                        .levels = 4,
                                                        .expectedL2Convergence = {1.0, 1.0, 1.0, 1.0},
                                                        .expectedLInfConvergence = {1.0, 1.0, 1.0, 1.0}},
                    (CompressibleFlowMmsTestParameters){.mpiTestParameter = {.testName = "low speed ausmpup", .nproc = 1, .arguments = "-dm_plex_separate_marker"},
                                                        .fluxCalculator = std::make_shared<ablate::flow::fluxCalculator::AusmpUp>(.3),

                                                        .constants = {.dim = 2,
                                                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                                                      .u = {.phiO = 70, .phiX = 5, .phiY = -7, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                                                      .v = {.phiO = 90, .phiX = -15, .phiY = -8.5, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                                                      .L = 1.0,
                                                                      .gamma = 1.4,
                                                                      .R = 287.0,
                                                                      .mu = 0.0,
                                                                      .k = 0.0},
                                                        .initialNx = 16,
                                                        .levels = 4,
                                                        .expectedL2Convergence = {1.0, 1.0, 1.2, 1.0},
                                                        .expectedLInfConvergence = {1.0, 1.0, 1.2, 1.0}},
                    (CompressibleFlowMmsTestParameters){.mpiTestParameter = {.testName = "high speed ausmpup", .nproc = 1, .arguments = "-dm_plex_separate_marker "},
                                                        .fluxCalculator = std::make_shared<ablate::flow::fluxCalculator::AusmpUp>(.3),

                                                        .constants = {.dim = 2,
                                                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                                                      .u = {.phiO = 800, .phiX = 50, .phiY = -30.0, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                                                      .v = {.phiO = 800, .phiX = -75, .phiY = 40, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                                                      .L = 1.0,
                                                                      .gamma = 1.4,
                                                                      .R = 287.0,
                                                                      .mu = 0.0,
                                                                      .k = 0.0},
                                                        .initialNx = 16,
                                                        .levels = 4,
                                                        .expectedL2Convergence = {1.0, 1.0, 1.0, 1.0},
                                                        .expectedLInfConvergence = {1.0, 1.0, 1.0, 1.0}},
                    (CompressibleFlowMmsTestParameters){
                        .mpiTestParameter = {.testName = "low speed ausm leastsquares", .nproc = 1, .arguments = "-dm_plex_separate_marker  -eulerpetscfv_type leastsquares"},
                        .fluxCalculator = std::make_shared<ablate::flow::fluxCalculator::Ausm>(),

                        .constants = {.dim = 2,
                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                      .u = {.phiO = 70, .phiX = 5, .phiY = -7, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                      .v = {.phiO = 90, .phiX = -15, .phiY = -8.5, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                      .L = 1.0,
                                      .gamma = 1.4,
                                      .R = 287.0,
                                      .mu = 0.0,
                                      .k = 0.0},
                        .initialNx = 16,
                        .levels = 4,
                        .expectedL2Convergence = {1.5, 1.5, 1.5, 1.5},
                        .expectedLInfConvergence = {1.0, 1.0, 1.0, 1.0}},
                    (CompressibleFlowMmsTestParameters){
                        .mpiTestParameter = {.testName = "high speed ausm leastsquares", .nproc = 1, .arguments = "-dm_plex_separate_marker -eulerpetscfv_type leastsquares"},
                        .fluxCalculator = std::make_shared<ablate::flow::fluxCalculator::Ausm>(),

                        .constants = {.dim = 2,
                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                      .u = {.phiO = 800, .phiX = 50, .phiY = -30.0, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                      .v = {.phiO = 800, .phiX = -75, .phiY = 40, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                      .L = 1.0,
                                      .gamma = 1.4,
                                      .R = 287.0,
                                      .mu = 0.0,
                                      .k = 0.0},
                        .initialNx = 16,
                        .levels = 4,
                        .expectedL2Convergence = {1.5, 1.5, 1.5, 1.5},
                        .expectedLInfConvergence = {1.0, 0.5, 1.0, 1.0}},
                    (CompressibleFlowMmsTestParameters){.mpiTestParameter = {.testName = "low speed average with conduction",
                                                                             .nproc = 1,
                                                                             .arguments = "-dm_plex_separate_marker -Tpetscfv_type leastsquares -velpetscfv_type leastsquares -petsclimiter_type none"},
                                                        .fluxCalculator = std::make_shared<ablate::flow::fluxCalculator::AverageFlux>(),

                                                        .constants = {.dim = 2,
                                                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                                                      .u = {.phiO = 70, .phiX = 5, .phiY = -7, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                                                      .v = {.phiO = 90, .phiX = -15, .phiY = -8.5, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                                                      .L = 1.0,
                                                                      .gamma = 1.4,
                                                                      .R = 287.0,
                                                                      .mu = 0.0,
                                                                      .k = 1000.0},
                                                        .initialNx = 4,
                                                        .levels = 4,
                                                        .expectedL2Convergence = {2, 2, 2, 2},
                                                        .expectedLInfConvergence = {1.9, 1.8, 1.8, 1.8}},
                    (CompressibleFlowMmsTestParameters){
                        .mpiTestParameter = {.testName = "high speed average with conduction",
                                             .nproc = 1,
                                             .arguments = "-dm_plex_separate_marker -Tpetscfv_type leastsquares -velpetscfv_type leastsquares -petsclimiter_type none "},
                        .fluxCalculator = std::make_shared<ablate::flow::fluxCalculator::AverageFlux>(),

                        .constants = {.dim = 2,
                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                      .u = {.phiO = 800, .phiX = 50, .phiY = -30.0, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                      .v = {.phiO = 800, .phiX = -75, .phiY = 40, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                      .L = 1.0,
                                      .gamma = 1.4,
                                      .R = 287.0,
                                      .mu = 0.0,
                                      .k = 1000.0},
                        .initialNx = 4,
                        .levels = 4,
                        .expectedL2Convergence = {2, 2, 2, 2},
                        .expectedLInfConvergence = {1.9, 1.8, 1.8, 1.8}},
                    (CompressibleFlowMmsTestParameters){.mpiTestParameter = {.testName = "low speed average with conduction and diffusion",
                                                                             .nproc = 1,
                                                                             .arguments = "-dm_plex_separate_marker -Tpetscfv_type leastsquares -velpetscfv_type leastsquares -petsclimiter_type none"},
                                                        .fluxCalculator = std::make_shared<ablate::flow::fluxCalculator::AverageFlux>(),

                                                        .constants = {.dim = 2,
                                                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                                                      .u = {.phiO = 70, .phiX = 5, .phiY = -7, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                                                      .v = {.phiO = 90, .phiX = -15, .phiY = -8.5, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                                                      .L = 1.0,
                                                                      .gamma = 1.4,
                                                                      .R = 287.0,
                                                                      .mu = 300.0,
                                                                      .k = 1000.0},
                                                        .initialNx = 4,
                                                        .levels = 4,
                                                        .expectedL2Convergence = {2, 2, 2, 2.2},
                                                        .expectedLInfConvergence = {1.9, 1.8, 1.8, 2.0}},
                    (CompressibleFlowMmsTestParameters){
                        .mpiTestParameter = {.testName = "high speed average with conduction and diffusion",
                                             .nproc = 1,
                                             .arguments = "-dm_plex_separate_marker -Tpetscfv_type leastsquares -velpetscfv_type leastsquares -petsclimiter_type none "},
                        .fluxCalculator = std::make_shared<ablate::flow::fluxCalculator::AverageFlux>(),

                        .constants = {.dim = 2,
                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                      .u = {.phiO = 800, .phiX = 50, .phiY = -30.0, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                      .v = {.phiO = 800, .phiX = -75, .phiY = 40, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                      .L = 1.0,
                                      .gamma = 1.4,
                                      .R = 287.0,
                                      .mu = 300.0,
                                      .k = 1000.0},
                        .initialNx = 4,
                        .levels = 4,
                        .expectedL2Convergence = {2, 2, 2, 2.0},
                        .expectedLInfConvergence = {1.9, 2.0, 1.8, 1.8}},
                    (CompressibleFlowMmsTestParameters){.mpiTestParameter = {.testName = "low speed average with conduction and diffusion 3D",
                                                                             .nproc = 1,
                                                                             .arguments = "-dm_plex_separate_marker -Tpetscfv_type leastsquares -velpetscfv_type leastsquares -petsclimiter_type none"},
                                                        .fluxCalculator = std::make_shared<ablate::flow::fluxCalculator::AverageFlux>(),

                                                        .constants = {.dim = 3,
                                                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = .4},
                                                                      .u = {.phiO = 70, .phiX = 5, .phiY = -7, .phiZ = 5, .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.5},
                                                                      .v = {.phiO = 90, .phiX = -15, .phiY = -8.5, .phiZ = 6.5, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.6},
                                                                      .w = {.phiO = 80, .phiX = -25, .phiY = 8.2, .phiZ = -10, .aPhiX = .75, .aPhiY = .2, .aPhiZ = 0.7},
                                                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.4e5, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.8},
                                                                      .L = 1.0,
                                                                      .gamma = 1.4,
                                                                      .R = 287.0,
                                                                      .mu = 300.0,
                                                                      .k = 1000.0},
                                                        .initialNx = 10,
                                                        .levels = 2,
                                                        .expectedL2Convergence = {2, 2.2, 2.2, 2.2, 2.},
                                                        .expectedLInfConvergence = {1.9, 2.2, 2.0, 2.0, 2.}}),
    [](const testing::TestParamInfo<CompressibleFlowMmsTestParameters> &info) { return info.param.mpiTestParameter.getTestName(); });
