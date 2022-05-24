#include <petsc.h>
#include "MpiTestFixture.hpp"
#include "domain/boxMesh.hpp"
#include "domain/modifiers/setFromOptions.hpp"
#include "environment/runEnvironment.hpp"
#include "finiteElement/boundaryConditions/essential.hpp"
#include "finiteElement/incompressibleFlow.h"
#include "finiteElement/incompressibleFlowSolver.hpp"
#include "finiteElement/lowMachFlowFields.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "parameters/mapParameters.hpp"
#include "parameters/petscOptionParameters.hpp"
#include "parameters/petscPrefixOptions.hpp"
#include "particles/inertial.hpp"
#include "particles/initializers/boxInitializer.hpp"
#include "solver/directSolverTsInterface.hpp"
#include "utilities/petscUtilities.hpp"

using namespace ablate;
using namespace ablate::finiteElement;

typedef PetscErrorCode (*ExactFunction)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);

typedef void (*IntegrandTestFunction)(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt *uOff, const PetscInt *uOff_x, const PetscScalar *u, const PetscScalar *u_t, const PetscScalar *u_x,
                                      const PetscInt *aOff, const PetscInt *aOff_x, const PetscScalar *a, const PetscScalar *a_t, const PetscScalar *a_x, PetscReal t, const PetscReal *X,
                                      PetscInt numConstants, const PetscScalar *constants, PetscScalar *f0);

#define SourceFunction(FUNC)            \
    FUNC(PetscInt dim,                  \
         PetscInt Nf,                   \
         PetscInt NfAux,                \
         const PetscInt uOff[],         \
         const PetscInt uOff_x[],       \
         const PetscScalar u[],         \
         const PetscScalar u_t[],       \
         const PetscScalar u_x[],       \
         const PetscInt aOff[],         \
         const PetscInt aOff_x[],       \
         const PetscScalar a[],         \
         const PetscScalar a_t[],       \
         const PetscScalar a_x[],       \
         PetscReal t,                   \
         const PetscReal X[],           \
         PetscInt numConstants,         \
         const PetscScalar constants[], \
         PetscScalar f0[])

// store the pointer to the provided test function from the solver
static IntegrandTestFunction f0_v_original;
static IntegrandTestFunction f0_w_original;
static IntegrandTestFunction f0_q_original;

struct ExactSolutionParameters {
    PetscInt dim;
    std::vector<PetscReal> pVel;
    PetscReal dp;
    PetscReal rhoP;
    PetscReal rhoF;
    PetscReal muF;
    PetscReal grav;
};

struct InertialParticleExactParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    ExactFunction uExact;
    ExactFunction pExact;
    ExactFunction TExact;
    ExactFunction u_tExact;
    ExactFunction T_tExact;
    ExactFunction particleExact;
    IntegrandTestFunction f0_v;
    IntegrandTestFunction f0_w;
    IntegrandTestFunction f0_q;
    ExactSolutionParameters parameters;
    std::shared_ptr<ablate::particles::initializers::Initializer> particleInitializer;
};

class InertialParticleExactTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<InertialParticleExactParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

static PetscErrorCode settling(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *x, void *ctx) {
    const PetscReal x0 = X[0];
    const PetscReal y0 = X[1];

    ExactSolutionParameters *parameters = (ExactSolutionParameters *)ctx;

    PetscReal tauP = parameters->rhoP * parameters->dp * parameters->dp / (18.0 * parameters->muF);  // particle relaxation time
    PetscReal uSt = tauP * parameters->grav * (1.0 - parameters->rhoF / parameters->rhoP);           // particle terminal (settling) velocity
    x[0] = uSt * (time + tauP * PetscExpReal(-time / tauP) - tauP) + x0;
    x[1] = y0;
    x[2] = uSt * (1.0 - PetscExpReal(-time / tauP));
    x[3] = 0.0;
    return 0;
}

static PetscErrorCode quiescent_u(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscInt d;
    for (d = 0; d < Dim; ++d) u[d] = 0.0;
    return 0;
}

static PetscErrorCode quiescent_u_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscInt d;
    for (d = 0; d < Dim; ++d) u[d] = 0.0;
    return 0;
}

static PetscErrorCode quiescent_p(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx) {
    p[0] = 0.0;
    return 0;
}

static PetscErrorCode quiescent_T(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = 0.0;
    return 0;
}
static PetscErrorCode quiescent_T_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = 0.0;
    return 0;
}

static void SourceFunction(f0_quiescent_v) {
    f0_v_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);
    f0[0] = 0;
    f0[1] = 0;
}

static void SourceFunction(f0_quiescent_w) {
    f0_w_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);
    f0[0] = 0;
}

static PetscErrorCode MonitorFlowAndParticleError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx) {
    PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    void *ctxs[3];
    DM dm;
    PetscDS ds;
    PetscReal ferrors[3];
    PetscInt f;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = DMGetDS(dm, &ds);
    CHKERRQ(ierr);

    // compute the flow error
    for (f = 0; f < 3; ++f) {
        ierr = PetscDSGetExactSolution(ds, f, &exactFuncs[f], &ctxs[f]);  // exsatFuncs are output
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
    }
    ierr = DMComputeL2FieldDiff(dm, crtime, exactFuncs, ctxs, u, ferrors);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // get the particle data from the context
    ablate::particles::Inertial *tracerParticles = (ablate::particles::Inertial *)ctx;
    PetscInt particleCount;
    ierr = DMSwarmGetSize(tracerParticles->GetParticleDM(), &particleCount);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // compute the average particle location
    const PetscReal *coords;
    PetscInt dims;
    PetscReal avg[3] = {0.0, 0.0, 0.0};
    ierr = DMSwarmGetField(tracerParticles->GetParticleDM(), DMSwarmPICField_coor, &dims, NULL, (void **)&coords);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    for (PetscInt n = 0; n < dims; n++) {
        for (PetscInt p = 0; p < particleCount; p++) {
            avg[n] += coords[p * dims + n] / PetscReal(particleCount);
        }
    }
    ierr = DMSwarmRestoreField(tracerParticles->GetParticleDM(), DMSwarmPICField_coor, &dims, NULL, (void **)&coords);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Timestep: %04d time = %-8.4g \t L_2 Error: [%2.3g, %2.3g, %2.3g] ParticleCount: %d\n",
                       (int)step,
                       (double)crtime,
                       (double)ferrors[0],
                       (double)ferrors[1],
                       (double)ferrors[2],
                       particleCount);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Avg Particle Location: [%2.3g, %2.3g, %2.3g]\n", (double)avg[0], (double)avg[1], (double)avg[2]);

    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    PetscFunctionReturn(0);
}

TEST_P(InertialParticleExactTestFixture, ParticleShouldMoveAsExpected) {
    StartWithMPI
        {
            TS ts; /* timestepper */
            // Get the testing param
            auto testingParam = GetParam();

            // initialize petsc and mpi
            ablate::environment::RunEnvironment::Initialize(argc, argv);
            ablate::utilities::PetscUtilities::Initialize();

            // setup the required fields for the flow
            std::vector<std::shared_ptr<domain::FieldDescriptor>> fieldDescriptors = {std::make_shared<ablate::finiteElement::LowMachFlowFields>()};

            // setup the ts
            TSCreate(PETSC_COMM_WORLD, &ts) >> testErrorChecker;
            auto mesh = std::make_shared<ablate::domain::BoxMesh>("mesh",
                                                                  fieldDescriptors,
                                                                  std::vector<std::shared_ptr<domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::SetFromOptions>()},
                                                                  std::vector<int>{2, 2},
                                                                  std::vector<double>{0.0, 0.0},
                                                                  std::vector<double>{1.0, 1.0});

            TSSetDM(ts, mesh->GetDM()) >> testErrorChecker;
            TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP) >> testErrorChecker;

            // Setup the flow data
            auto parameters = std::make_shared<ablate::parameters::PetscOptionParameters>();

            auto velocityExact = std::make_shared<mathFunctions::FieldFunction>(
                "velocity", ablate::mathFunctions::Create(testingParam.uExact, &testingParam.parameters), ablate::mathFunctions::Create(testingParam.u_tExact, &testingParam.parameters));
            auto pressureExact = std::make_shared<mathFunctions::FieldFunction>("pressure", ablate::mathFunctions::Create(testingParam.pExact, &testingParam.parameters));
            auto temperatureExact = std::make_shared<mathFunctions::FieldFunction>(
                "temperature", ablate::mathFunctions::Create(testingParam.TExact, &testingParam.parameters), ablate::mathFunctions::Create(testingParam.T_tExact, &testingParam.parameters));

            auto flowObject = std::make_shared<ablate::finiteElement::IncompressibleFlowSolver>(
                "testFlow",
                domain::Region::ENTIREDOMAIN,
                nullptr,
                parameters,
                /* boundary conditions */
                std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>>{std::make_shared<boundaryConditions::Essential>("wall velocity", std::vector<int>{3, 1, 2, 4}, velocityExact),
                                                                                    std::make_shared<boundaryConditions::Essential>("wall temp", std::vector<int>{3, 1, 2, 4}, temperatureExact)},
                /* aux updates*/
                std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{});

            auto particleParameters = std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"fluidDensity", std::to_string(testingParam.parameters.rhoF)},
                                                                                                                             {"fluidViscosity", std::to_string(testingParam.parameters.muF)},
                                                                                                                             {"gravityField", std::to_string(testingParam.parameters.grav) + " 0 0"}});

            // convert the constant values to fieldInitializations
            auto fieldInitialization = std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{
                std::make_shared<mathFunctions::FieldFunction>(ablate::particles::Inertial::ParticleVelocity, ablate::mathFunctions::Create(testingParam.parameters.pVel)),
                std::make_shared<mathFunctions::FieldFunction>(ablate::particles::Inertial::ParticleDiameter, ablate::mathFunctions::Create(testingParam.parameters.dp)),
                std::make_shared<mathFunctions::FieldFunction>(ablate::particles::Inertial::ParticleDensity, ablate::mathFunctions::Create(testingParam.parameters.rhoP)),
            };

            // Use the petsc options that start with -particle_
            auto particleOptions = std::make_shared<ablate::parameters::PetscPrefixOptions>("-particle_");

            // store the exact solution
            auto exactSolutionFunction = ablate::mathFunctions::Create(testingParam.particleExact, &testingParam.parameters);

            // Create an inertial particle object
            auto particles = std::make_shared<ablate::particles::Inertial>(
                "particle", ablate::domain::Region::ENTIREDOMAIN, particleOptions, 2, particleParameters, GetParam().particleInitializer, fieldInitialization, exactSolutionFunction);

            mesh->InitializeSubDomains({flowObject, particles},
                                       std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{velocityExact, pressureExact, temperatureExact},
                                       /* exact solutions*/
                                       std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{velocityExact, pressureExact, temperatureExact});
            solver::DirectSolverTsInterface directSolverTsInterface(ts, {flowObject, particles});

            // Override problem with source terms, boundary, and set the exact solution
            {
                PetscDS prob;
                DMGetDS(mesh->GetDM(), &prob) >> testErrorChecker;

                // V, W Test Function
                IntegrandTestFunction tempFunctionPointer;
                if (testingParam.f0_v) {
                    PetscDSGetResidual(prob, VTEST, &f0_v_original, &tempFunctionPointer) >> testErrorChecker;
                    PetscDSSetResidual(prob, VTEST, testingParam.f0_v, tempFunctionPointer) >> testErrorChecker;
                }
                if (testingParam.f0_w) {
                    PetscDSGetResidual(prob, WTEST, &f0_w_original, &tempFunctionPointer) >> testErrorChecker;
                    PetscDSSetResidual(prob, WTEST, testingParam.f0_w, tempFunctionPointer) >> testErrorChecker;
                }
                if (testingParam.f0_q) {
                    PetscDSGetResidual(prob, QTEST, &f0_q_original, &tempFunctionPointer) >> testErrorChecker;
                    PetscDSSetResidual(prob, QTEST, testingParam.f0_q, tempFunctionPointer) >> testErrorChecker;
                }
            }

            // Check the convergence
            DMTSCheckFromOptions(ts, mesh->GetSolutionVector()) >> testErrorChecker;

            TSSetComputeInitialCondition(particles->GetTS(), ablate::particles::Particles::ComputeParticleExactSolution) >> testErrorChecker;

            // setup the flow monitor to also check particles
            TSMonitorSet(ts, MonitorFlowAndParticleError, particles.get(), NULL) >> testErrorChecker;
            TSSetFromOptions(ts) >> testErrorChecker;

            // Solve the one way coupled system
            TSSolve(ts, mesh->GetSolutionVector()) >> testErrorChecker;

            // Compare the actual vs expected values
            DMTSCheckFromOptions(ts, mesh->GetSolutionVector()) >> testErrorChecker;

            // Cleanup
            TSDestroy(&ts) >> testErrorChecker;
        }
        ablate::environment::RunEnvironment::Finalize();
        exit(0);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(InertialParticleTests, InertialParticleExactTestFixture,
                         testing::Values((InertialParticleExactParameters){.mpiTestParameter = {.testName = "single inertial particle settling in quiescent fluid",
                                                                                                .nproc = 1,
                                                                                                .expectedOutputFile = "outputs/particles/inertialParticle_settling_in_quiescent_fluid_single",
                                                                                                .arguments = "-dm_plex_separate_marker -dm_refine 2 "
                                                                                                             "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                                                                                                             "-dmts_check .001 -ts_max_steps 7 -ts_dt 0.06 -ksp_type fgmres -ksp_gmres_restart 10 "
                                                                                                             "-ksp_rtol 1.0e-9 -ksp_error_if_not_converged -pc_type fieldsplit  "
                                                                                                             " -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                                                                                             "-particle_ts_dt 0.03 -particle_ts_convergence_estimate -convest_num_refine 1 "},
                                                                           .uExact = quiescent_u,
                                                                           .pExact = quiescent_p,
                                                                           .TExact = quiescent_T,
                                                                           .u_tExact = quiescent_u_t,
                                                                           .T_tExact = quiescent_T_t,
                                                                           .particleExact = settling,
                                                                           .f0_v = f0_quiescent_v,
                                                                           .f0_w = f0_quiescent_w,
                                                                           .parameters = {.dim = 2, .pVel = {0.0, 0.0}, .dp = 0.22, .rhoP = 90.0, .rhoF = 1.0, .muF = 1.0, .grav = 1.0},
                                                                           .particleInitializer = std::make_shared<ablate::particles::initializers::BoxInitializer>(std::vector<double>{0.5, 0.5},
                                                                                                                                                                    std::vector<double>{.5, .5}, 1)},
                                         (InertialParticleExactParameters){
                                             .mpiTestParameter = {.testName = "multi inertial particle settling in quiescent fluid",
                                                                  .nproc = 1,
                                                                  .expectedOutputFile = "outputs/particles/inertialParticle_settling_in_quiescent_fluid_multi",
                                                                  .arguments = "-dm_plex_separate_marker -dm_refine 2 "
                                                                               "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                                                                               "-dmts_check .001 -ts_max_steps 7 -ts_dt 0.06 -ksp_type fgmres -ksp_gmres_restart 10 "
                                                                               "-ksp_rtol 1.0e-9 -ksp_error_if_not_converged -pc_type fieldsplit  "
                                                                               " -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                                                               "-particle_ts_dt 0.03 -particle_ts_convergence_estimate -convest_num_refine 1 "},
                                             .uExact = quiescent_u,
                                             .pExact = quiescent_p,
                                             .TExact = quiescent_T,
                                             .u_tExact = quiescent_u_t,
                                             .T_tExact = quiescent_T_t,
                                             .particleExact = settling,
                                             .f0_v = f0_quiescent_v,
                                             .f0_w = f0_quiescent_w,
                                             .parameters = {.dim = 2, .pVel = {0.0, 0.0}, .dp = 0.22, .rhoP = 90.0, .rhoF = 1.0, .muF = 1.0, .grav = 1.0},
                                             .particleInitializer = std::make_shared<ablate::particles::initializers::BoxInitializer>(std::vector<double>{0.2, 0.3}, std::vector<double>{.4, .6}, 10)},
                                         (InertialParticleExactParameters){.mpiTestParameter = {.testName = "deletion inertial particles settling in quiescent fluid",
                                                                                                .nproc = 1,
                                                                                                .expectedOutputFile = "outputs/particles/inertialParticles_settling_in_quiescent_fluid_deletion",
                                                                                                .arguments = "-dm_plex_separate_marker -dm_refine 2 "
                                                                                                             "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                                                                                                             "-dmts_check .001 -ts_max_steps 7 -ts_dt 0.06 -ksp_type fgmres -ksp_gmres_restart 10 "
                                                                                                             "-ksp_rtol 1.0e-9 -ksp_error_if_not_converged -pc_type fieldsplit  "
                                                                                                             " -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                                                                                             "-particle_ts_dt 0.03 -particle_ts_convergence_estimate -convest_num_refine 1 "},
                                                                           .uExact = quiescent_u,
                                                                           .pExact = quiescent_p,
                                                                           .TExact = quiescent_T,
                                                                           .u_tExact = quiescent_u_t,
                                                                           .T_tExact = quiescent_T_t,
                                                                           .particleExact = settling,
                                                                           .f0_v = f0_quiescent_v,
                                                                           .f0_w = f0_quiescent_w,
                                                                           .parameters = {.dim = 2, .pVel = {0.0, 0.0}, .dp = 0.22, .rhoP = 90.0, .rhoF = 1.0, .muF = 1.0, .grav = 1.0},
                                                                           .particleInitializer = std::make_shared<ablate::particles::initializers::BoxInitializer>(std::vector<double>{0.92, 0.3},
                                                                                                                                                                    std::vector<double>{.98, .6}, 10)}),
                         [](const testing::TestParamInfo<InertialParticleExactParameters> &info) { return info.param.mpiTestParameter.getTestName(); });
