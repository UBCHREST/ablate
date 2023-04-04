#include <petsc.h>
#include <memory>
#include "MpiTestFixture.hpp"
#include "domain/boxMesh.hpp"
#include "environment/runEnvironment.hpp"
#include "finiteElement/boundaryConditions/essential.hpp"
#include "finiteElement/incompressibleFlow.h"
#include "finiteElement/incompressibleFlowSolver.hpp"
#include "finiteElement/lowMachFlowFields.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "parameters/petscOptionParameters.hpp"
#include "parameters/petscPrefixOptions.hpp"
#include "particles/initializers/boxInitializer.hpp"
#include "particles/particleSolver.hpp"
#include "particles/processes/tracer.hpp"
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
static PetscReal omega;

struct TracerParticleMMSParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    ExactFunction uExact;
    ExactFunction pExact;
    ExactFunction TExact;
    ExactFunction uDerivativeExact;
    ExactFunction pDerivativeExact;
    ExactFunction TDerivativeExact;
    ExactFunction particleExact;
    IntegrandTestFunction f0_v;
    IntegrandTestFunction f0_w;
    IntegrandTestFunction f0_q;
    PetscReal omega;
};

class TracerParticleMMSTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<TracerParticleMMSParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

/*
  CASE: trigonometric-trigonometric
  In 2D we use exact solution:

    x = r0 cos(w t + theta0)  r0     = sqrt(x0^2 + y0^2)
    y = r0 sin(w t + theta0)  theta0 = arctan(y0/x0)
    u = -w r0 sin(theta0) = -w y
    v =  w r0 cos(theta0) =  w x
    p = x + y - 1
    T = t + x + y
    f = <1, 1>
    Q = 1 + w (x - y)/r

  so that

    \nabla \cdot u = 0 + 0 = 0

  f = du/dt + u \cdot \nabla u - \nu \Delta u + \nabla p
    = <0, 0> + u_i d_i u_j - \nu 0 + <1, 1>
    = <1, 1> + w^2 <-y, x> . <<0, 1>, <-1, 0>>
    = <1, 1> + w^2 <-x, -y>
    = <1, 1> - w^2 <x, y>

  Q = dT/dt + u \cdot \nabla T - \alpha \Delta T
    = 1 + <u, v> . <1, 1> - \alpha 0
    = 1 + u + v
*/
static PetscErrorCode trig_trig_x(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *x, void *ctx) {
    const PetscReal x0 = X[0];
    const PetscReal y0 = X[1];
    const PetscReal R0 = PetscSqrtReal(x0 * x0 + y0 * y0);
    const PetscReal theta0 = PetscAtan2Real(y0, x0);

    x[0] = R0 * PetscCosReal(omega * time + theta0);
    x[1] = R0 * PetscSinReal(omega * time + theta0);
    return 0;
}
static PetscErrorCode trig_trig_u(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = -omega * X[1];
    u[1] = omega * X[0];
    return 0;
}
static PetscErrorCode trig_trig_u_t(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = 0.0;
    u[1] = 0.0;
    return 0;
}

static PetscErrorCode trig_trig_p(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx) {
    p[0] = X[0] + X[1] - 1.0;
    return 0;
}

static PetscErrorCode trig_trig_p_t(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx) {
    p[0] = 0.0;
    return 0;
}

static PetscErrorCode trig_trig_T(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = time + X[0] + X[1];
    return 0;
}
static PetscErrorCode trig_trig_T_t(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = 1.0;
    return 0;
}

static void SourceFunction(f0_trig_trig_v) {
    f0_v_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);
    f0[0] -= 1.0 - omega * omega * X[0];
    f0[1] -= 1.0 - omega * omega * X[1];
}

static void SourceFunction(f0_trig_trig_w) {
    f0_w_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);

    f0[0] += -(1.0 + omega * (X[0] - X[1]));
}

/*
  CASE: linear particle movement
  In 2D we use exact solution:

    x = t + xo
    y = t*t/2 + t*xo + yo
    u = 1
    v = x
    p = x + y - 1
    T = t + x + y

  so that

    \nabla \cdot u = 0 + 0 = 0

  // see docs/content/formulations/incompressibleFlow/solutions/Incompressible_2D_Linear_MMS.nb
*/
static PetscErrorCode linear_x(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *x, void *ctx) {
    const PetscReal x0 = X[0];
    const PetscReal y0 = X[1];

    x[0] = time + x0;
    x[1] = time * time / 2 + time * x0 + y0;
    return 0;
}
static PetscErrorCode linear_u(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = 1.0;
    u[1] = X[0];
    return 0;
}
static PetscErrorCode linear_u_t(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = 0.0;
    u[1] = 0.0;
    return 0;
}

static PetscErrorCode linear_p(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx) {
    p[0] = X[0] + X[1] - 1.0;
    return 0;
}

static PetscErrorCode linear_p_t(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx) {
    p[0] = 0.0;
    return 0;
}

static PetscErrorCode linear_T(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = time + X[0] + X[1];
    return 0;
}
static PetscErrorCode linear_T_t(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = 1.0;
    return 0;
}

static void SourceFunction(f0_linear_v) {
    f0_v_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);

    const PetscReal rho = 1.0;

    f0[0] -= 1;
    f0[1] -= 1 + rho;
}

static void SourceFunction(f0_linear_w) {
    f0_w_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);

    const PetscReal rho = 1.0;
    const PetscReal S = constants[STROUHAL];
    const PetscReal Cp = constants[CP];

    f0[0] -= Cp * rho * (1 + S + X[0]);
}

static PetscErrorCode MonitorFlowAndParticleError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx) {
    PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    void *ctxs[3];
    DM dm;
    PetscDS ds;
    PetscReal ferrors[3];
    PetscInt f;

    PetscFunctionBeginUser;
    PetscCallAbort(PETSC_COMM_WORLD, TSGetDM(ts, &dm));
    PetscCallAbort(PETSC_COMM_WORLD, DMGetDS(dm, &ds));

    // compute the flow error
    for (f = 0; f < 3; ++f) {
        PetscCallAbort(PETSC_COMM_WORLD, PetscDSGetExactSolution(ds, f, &exactFuncs[f], &ctxs[f]));
    }
    PetscCallAbort(PETSC_COMM_WORLD, DMComputeL2FieldDiff(dm, crtime, exactFuncs, ctxs, u, ferrors));

    // get the particle data from the context
    auto *tracerParticles = (ablate::particles::ParticleSolver *)ctx;
    PetscInt particleCount;
    PetscCallAbort(PETSC_COMM_WORLD, DMSwarmGetSize(tracerParticles->GetParticleDM(), &particleCount));

    // compute the average particle location
    const PetscReal *coords;
    PetscInt dims;
    PetscReal avg[3] = {0.0, 0.0, 0.0};
    PetscCallAbort(PETSC_COMM_WORLD, DMSwarmGetField(tracerParticles->GetParticleDM(), DMSwarmPICField_coor, &dims, NULL, (void **)&coords));
    for (PetscInt p = 0; p < particleCount; p++) {
        for (PetscInt n = 0; n < dims; n++) {
            avg[n] += coords[p * dims + n] / PetscReal(particleCount);
        }
    }
    PetscCallAbort(PETSC_COMM_WORLD, DMSwarmRestoreField(tracerParticles->GetParticleDM(), DMSwarmPICField_coor, &dims, NULL, (void **)&coords));

    PetscCallAbort(PETSC_COMM_WORLD,
                   PetscPrintf(PETSC_COMM_WORLD,
                               "Timestep: %04" PetscInt_FMT " time = %-8.4g \t L_2 Error: [%2.3g, %2.3g, %2.3g] ParticleCount: %" PetscInt_FMT "\n",
                               step,
                               (double)crtime,
                               (double)ferrors[0],
                               (double)ferrors[1],
                               (double)ferrors[2],
                               particleCount));
    PetscCallAbort(PETSC_COMM_WORLD, PetscPrintf(PETSC_COMM_WORLD, "Avg Particle Location: [%2.3g, %2.3g, %2.3g]\n", (double)avg[0], (double)avg[1], (double)avg[2]));
    PetscFunctionReturn(0);
}

TEST_P(TracerParticleMMSTestFixture, ParticleTracerFlowMMSTests) {
    StartWithMPI
        {
            // Get the testing param
            auto testingParam = GetParam();
            omega = testingParam.omega;

            // initialize petsc and mpi
            ablate::environment::RunEnvironment::Initialize(argc, argv);
            ablate::utilities::PetscUtilities::Initialize();

            // setup the required fields for the flow
            std::vector<std::shared_ptr<domain::FieldDescriptor>> fieldDescriptors = {std::make_shared<ablate::finiteElement::LowMachFlowFields>()};

            // setup the mesh
            auto mesh = std::make_shared<ablate::domain::BoxMesh>(
                "mesh", fieldDescriptors, std::vector<std::shared_ptr<domain::modifiers::Modifier>>{}, std::vector<int>{2, 2}, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0});

            // Setup the flow data
            // pull the parameters from the petsc options
            auto parameters = std::make_shared<ablate::parameters::PetscOptionParameters>();

            auto velocityExact =
                std::make_shared<mathFunctions::FieldFunction>("velocity", ablate::mathFunctions::Create(testingParam.uExact), ablate::mathFunctions::Create(testingParam.uDerivativeExact));
            auto pressureExact =
                std::make_shared<mathFunctions::FieldFunction>("pressure", ablate::mathFunctions::Create(testingParam.pExact), ablate::mathFunctions::Create(testingParam.pDerivativeExact));
            auto temperatureExact =
                std::make_shared<mathFunctions::FieldFunction>("temperature", ablate::mathFunctions::Create(testingParam.TExact), ablate::mathFunctions::Create(testingParam.TDerivativeExact));

            // create a time stepper
            auto timeStepper = ablate::solver::TimeStepper(mesh,
                                                           nullptr,
                                                           {},
                                                           std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{velocityExact, pressureExact, temperatureExact},
                                                           std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{velocityExact, pressureExact, temperatureExact});

            timeStepper.Register(std::make_shared<ablate::finiteElement::IncompressibleFlowSolver>(
                "testFlow",
                domain::Region::ENTIREDOMAIN,
                nullptr,
                parameters,
                /* boundary conditions */
                std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>>{std::make_shared<boundaryConditions::Essential>("top wall velocity", 3, velocityExact),
                                                                                    std::make_shared<boundaryConditions::Essential>("bottom wall velocity", 1, velocityExact),
                                                                                    std::make_shared<boundaryConditions::Essential>("right wall velocity", 2, velocityExact),
                                                                                    std::make_shared<boundaryConditions::Essential>("left wall velocity", 4, velocityExact),
                                                                                    std::make_shared<boundaryConditions::Essential>("top wall temp", 3, temperatureExact),
                                                                                    std::make_shared<boundaryConditions::Essential>("bottom wall temp", 1, temperatureExact),
                                                                                    std::make_shared<boundaryConditions::Essential>("right wall temp", 2, temperatureExact),
                                                                                    std::make_shared<boundaryConditions::Essential>("left wall temp", 4, temperatureExact)},
                /* aux updates*/
                std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{}));

            // Create the particle domain
            // pass all options with the particles prefix to the particle object
            auto particleOptions = std::make_shared<ablate::parameters::PetscPrefixOptions>("-particle_");
            auto initializer = std::make_shared<ablate::particles::initializers::BoxInitializer>(std::vector<double>{0.25, 0.25}, std::vector<double>{.75, .75}, 5);
            auto particles =
                std::make_shared<ablate::particles::ParticleSolver>("particle",
                                                                    ablate::domain::Region::ENTIREDOMAIN,
                                                                    particleOptions,
                                                                    std::vector<ablate::particles::FieldDescription>{},
                                                                    std::vector<std::shared_ptr<ablate::particles::processes::Process>>{std::make_shared<ablate::particles::processes::Tracer>()},
                                                                    initializer,
                                                                    /**
                                                                     * no fields to initialize
                                                                     */
                                                                    std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{},
                                                                    /** exact solutions**/
                                                                    std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{std::make_shared<ablate::mathFunctions::FieldFunction>(
                                                                        particles::ParticleSolver::ParticleCoordinates, ablate::mathFunctions::Create(testingParam.particleExact))});
            timeStepper.Register(particles);

            // Setup the solvers
            timeStepper.Initialize();

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
            DMTSCheckFromOptions(timeStepper.GetTS(), mesh->GetSolutionVector()) >> testErrorChecker;

            // setup the initial conditions for error computing, this is only used for tests
            TSSetComputeInitialCondition(particles->GetParticleTS(), ablate::particles::ParticleSolver::ComputeParticleExactSolution) >> testErrorChecker;

            // Solve the one way coupled system
            TSMonitorSet(timeStepper.GetTS(), MonitorFlowAndParticleError, particles.get(), NULL) >> testErrorChecker;
            TSSetFromOptions(timeStepper.GetTS()) >> testErrorChecker;
            timeStepper.Solve();

            // Compare the actual vs expected values
            DMTSCheckFromOptions(timeStepper.GetTS(), mesh->GetSolutionVector()) >> testErrorChecker;
        }
        ablate::environment::RunEnvironment::Finalize();
        exit(0);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(TracerParticleTests, TracerParticleMMSTestFixture,
                         testing::Values((TracerParticleMMSParameters){.mpiTestParameter = testingResources::MpiTestParameter(
                                                                           "particle in incompressible 2d trigonometric trigonometric tri_p2_p1_p1", 1,
                                                                           "-dm_plex_separate_marker -dm_refine 2 -vel_petscspace_degree 2 -pres_petscspace_degree 1 "
                                                                           "-temp_petscspace_degree 1 -dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 -ts_monitor_cancel "
                                                                           "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                                                                           "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 "
                                                                           "-pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_0_pc_type lu "
                                                                           "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi "
                                                                           "-particle_ts_dt 0.05 -particle_ts_convergence_estimate -convest_num_refine 1 "
                                                                           "-particle_ts_monitor_cancel",
                                                                           "outputs/particles/tracerParticles_incompressible_trigonometric_2d_tri_p2_p1_p1"),
                                                                       .uExact = trig_trig_u,
                                                                       .pExact = trig_trig_p,
                                                                       .TExact = trig_trig_T,
                                                                       .uDerivativeExact = trig_trig_u_t,
                                                                       .pDerivativeExact = trig_trig_p_t,
                                                                       .TDerivativeExact = trig_trig_T_t,
                                                                       .particleExact = trig_trig_x,
                                                                       .f0_v = f0_trig_trig_v,
                                                                       .f0_w = f0_trig_trig_w,
                                                                       .f0_q = nullptr,
                                                                       .omega = 0.5},
                                         (TracerParticleMMSParameters){
                                             .mpiTestParameter = testingResources::MpiTestParameter("particle deletion with simple fluid tri_p2_p1_p1", 1,
                                                                                                    "-dm_plex_separate_marker -dm_refine 2 "
                                                                                                    "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                                                                                                    "-dmts_check .001 -ts_max_steps 7 -ts_dt 0.06 -ksp_type fgmres -ksp_gmres_restart 10 "
                                                                                                    "-ksp_rtol 1.0e-9 -ksp_error_if_not_converged -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 "
                                                                                                    "-pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                                                                                    "-fieldsplit_0_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi "
                                                                                                    "-particle_ts_dt 0.03 -particle_ts_convergence_estimate -convest_num_refine 1 ",
                                                                                                    "outputs/particles/tracerParticles_deletion_with_simple_fluid_tri_p2_p1_p1"),
                                             .uExact = linear_u,
                                             .pExact = linear_p,
                                             .TExact = linear_T,
                                             .uDerivativeExact = linear_u_t,
                                             .pDerivativeExact = linear_p_t,
                                             .TDerivativeExact = linear_T_t,
                                             .particleExact = linear_x,
                                             .f0_v = f0_linear_v,
                                             .f0_w = f0_linear_w,
                                             .f0_q = nullptr,
                                             .omega = NAN}),
                         [](const testing::TestParamInfo<TracerParticleMMSParameters> &info) { return info.param.mpiTestParameter.getTestName(); });