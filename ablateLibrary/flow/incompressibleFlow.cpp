#include "incompressibleFlow.hpp"
#include "incompressibleFlow.h"
#include "utilities/petscError.hpp"

/*
  CASE: incompressible quadratic
  In 2D we use exact solution:

    u = t + x^2 + y^2
    v = t + 2x^2 - 2xy
    p = x + y - 1
    T = t + x + y
  so that

    \nabla \cdot u = 2x - 2x = 0

  see docs/content/formulations/incompressibleFlow/solutions/Incompressible_2D_Quadratic_MMS.nb
*/
static PetscErrorCode incompressible_quadratic_u(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = time + X[0] * X[0] + X[1] * X[1];
    u[1] = time + 2.0 * X[0] * X[0] - 2.0 * X[0] * X[1];
    return 0;
}
static PetscErrorCode incompressible_quadratic_u_t(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = 1.0;
    u[1] = 1.0;
    return 0;
}

static PetscErrorCode incompressible_quadratic_p(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *p, void *ctx) {
    p[0] = X[0] + X[1] - 1.0;
    return 0;
}

static PetscErrorCode incompressible_quadratic_T(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = time + X[0] + X[1];
    return 0;
}
static PetscErrorCode incompressible_quadratic_T_t(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = 1.0;
    return 0;
}

ablate::flow::IncompressibleFlow::IncompressibleFlow(std::shared_ptr<mesh::Mesh> mesh, std::string name, std::map<std::string, std::string> arguments,
                                                     std::shared_ptr<parameters::Parameters> parameters)
    : Flow(mesh, name, arguments) {
    // Setup the problem
    IncompressibleFlow_SetupDiscretization(mesh->GetDomain()) >> checkError;

    // Pack up any of the parameters
    PetscScalar constants[TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS];
    parameters->Fill(TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS, incompressibleFlowParametersTypeNames, constants);

    // Start the problem setup
    IncompressibleFlow_StartProblemSetup(mesh->GetDomain(), TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS, constants) >> checkError;

    // Apply any boundary conditions //TODO: move to seperate class
    PetscDS prob;
    DMGetDS(mesh->GetDomain(), &prob) >> checkError;

    PetscInt id;
    id = 3;
    PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall velocity", "marker", VEL, 0, NULL, (void (*)(void))incompressible_quadratic_u, (void (*)(void))incompressible_quadratic_u_t, 1, &id, nullptr) >>
        checkError;
    id = 1;
    PetscDSAddBoundary(
        prob, DM_BC_ESSENTIAL, "bottom wall velocity", "marker", VEL, 0, NULL, (void (*)(void))incompressible_quadratic_u, (void (*)(void))incompressible_quadratic_u_t, 1, &id, nullptr) >>
        checkError;
    id = 2;
    PetscDSAddBoundary(
        prob, DM_BC_ESSENTIAL, "right wall velocity", "marker", VEL, 0, NULL, (void (*)(void))incompressible_quadratic_u, (void (*)(void))incompressible_quadratic_u_t, 1, &id, nullptr) >>
        checkError;
    id = 4;
    PetscDSAddBoundary(
        prob, DM_BC_ESSENTIAL, "left wall velocity", "marker", VEL, 0, NULL, (void (*)(void))incompressible_quadratic_u, (void (*)(void))incompressible_quadratic_u_t, 1, &id, nullptr) >>
        checkError;
    id = 3;
    PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall temp", "marker", TEMP, 0, NULL, (void (*)(void))incompressible_quadratic_T, (void (*)(void))incompressible_quadratic_T_t, 1, &id, nullptr) >>
        checkError;
    id = 1;
    PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "bottom wall temp", "marker", TEMP, 0, NULL, (void (*)(void))incompressible_quadratic_T, (void (*)(void))incompressible_quadratic_T_t, 1, &id, nullptr) >>
        checkError;
    id = 2;
    PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall temp", "marker", TEMP, 0, NULL, (void (*)(void))incompressible_quadratic_T, (void (*)(void))incompressible_quadratic_T_t, 1, &id, nullptr) >>
        checkError;
    id = 4;
    PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall temp", "marker", TEMP, 0, NULL, (void (*)(void))incompressible_quadratic_T, (void (*)(void))incompressible_quadratic_T_t, 1, &id, nullptr) >>
        checkError;

    // Set the exact solution
    PetscDSSetExactSolution(prob, VEL, incompressible_quadratic_u, NULL) >> checkError;
    PetscDSSetExactSolution(prob, PRES, incompressible_quadratic_p, NULL) >> checkError;
    PetscDSSetExactSolution(prob, TEMP, incompressible_quadratic_T, NULL) >> checkError;
    PetscDSSetExactSolutionTimeDerivative(prob, VEL, incompressible_quadratic_u_t, NULL) >> checkError;
    PetscDSSetExactSolutionTimeDerivative(prob, PRES, NULL, NULL) >> checkError;
    PetscDSSetExactSolutionTimeDerivative(prob, TEMP, incompressible_quadratic_T_t, NULL) >> checkError;
}

Vec ablate::flow::IncompressibleFlow::SetupSolve(TS &ts) {
    // Setup the solve with the ts
    TSSetDM(ts, mesh->GetDomain()) >> checkError;

    // finish setup and assign flow field
    IncompressibleFlow_CompleteProblemSetup(ts, &flowSolution);

    // Initialize the flow field
    DMComputeExactSolution(mesh->GetDomain(), 0, flowSolution, NULL) >> checkError;

    // Name the flow field
    PetscObjectSetName((PetscObject)flowSolution, "Incompressible Flow Numerical Solution") >> checkError;
    VecSetOptionsPrefix(flowSolution, "num_sol_") >> checkError;

    // set the dm on the ts
    TSSetDM(ts, mesh->GetDomain()) >> checkError;

    return flowSolution;
}
