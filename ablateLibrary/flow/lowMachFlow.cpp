#include "lowMachFlow.hpp"
#include "lowMachFlow.h"
#include "utilities/petscError.hpp"
#include "parser/registrar.hpp"

static PetscErrorCode lowMach_quadratic_u(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    // u = {t + x^2 + y^2, t + 2*x^2 + 2*x*y}
    u[0] = time + X[0] * X[0] + X[1] * X[1];
    u[1] = time + 2.0 * X[0] * X[0] + 2.0 * X[0] * X[1];
    return 0;
}
static PetscErrorCode lowMach_quadratic_u_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = 1.0;
    u[1] = 1.0;
    return 0;
}

static PetscErrorCode lowMach_quadratic_p(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx) {
    // p = x + y - 1
    p[0] = X[0] + X[1] - 1.0;
    return 0;
}

static PetscErrorCode lowMach_quadratic_T(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    // T = t + x + y + 1
    T[0] = time + X[0] + X[1] + 1;
    return 0;
}
static PetscErrorCode lowMach_quadratic_T_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = 1.0;
    return 0;
}

ablate::flow::LowMachFlow::LowMachFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::map<std::string, std::string> arguments, std::shared_ptr<parameters::Parameters> parameters)
    : Flow(mesh, name, arguments) {
    // Setup the problem
    LowMachFlow_SetupDiscretization(mesh->GetDomain()) >> checkError;

    // Pack up any of the parameters
    PetscScalar constants[TOTAL_LOW_MACH_FLOW_PARAMETERS];
    parameters->Fill(TOTAL_LOW_MACH_FLOW_PARAMETERS, lowMachFlowParametersTypeNames, constants);

    // Start the problem setup
    LowMachFlow_StartProblemSetup(mesh->GetDomain(), TOTAL_LOW_MACH_FLOW_PARAMETERS, constants) >> checkError;

    // Apply any boundary conditions //TODO: move to seperate class
    PetscDS prob;
    DMGetDS(mesh->GetDomain(), &prob) >> checkError;

    PetscInt id;
    id = 3;
    PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall velocity", "marker", VEL, 0, NULL, (void (*)(void))lowMach_quadratic_u, (void (*)(void))lowMach_quadratic_u_t, 1, &id, nullptr) >> checkError;
    id = 1;
    PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "bottom wall velocity", "marker", VEL, 0, NULL, (void (*)(void))lowMach_quadratic_u, (void (*)(void))lowMach_quadratic_u_t, 1, &id, nullptr) >>
        checkError;
    id = 2;
    PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall velocity", "marker", VEL, 0, NULL, (void (*)(void))lowMach_quadratic_u, (void (*)(void))lowMach_quadratic_u_t, 1, &id, nullptr) >> checkError;
    id = 4;
    PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall velocity", "marker", VEL, 0, NULL, (void (*)(void))lowMach_quadratic_u, (void (*)(void))lowMach_quadratic_u_t, 1, &id, nullptr) >> checkError;
    id = 3;
    PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall temp", "marker", TEMP, 0, NULL, (void (*)(void))lowMach_quadratic_T, (void (*)(void))lowMach_quadratic_T_t, 1, &id, nullptr) >> checkError;
    id = 1;
    PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "bottom wall temp", "marker", TEMP, 0, NULL, (void (*)(void))lowMach_quadratic_T, (void (*)(void))lowMach_quadratic_T_t, 1, &id, nullptr) >> checkError;
    id = 2;
    PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall temp", "marker", TEMP, 0, NULL, (void (*)(void))lowMach_quadratic_T, (void (*)(void))lowMach_quadratic_T_t, 1, &id, nullptr) >> checkError;
    id = 4;
    PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall temp", "marker", TEMP, 0, NULL, (void (*)(void))lowMach_quadratic_T, (void (*)(void))lowMach_quadratic_T_t, 1, &id, nullptr) >> checkError;

    // Set the exact solution
    PetscDSSetExactSolution(prob, VEL, lowMach_quadratic_u, NULL) >> checkError;
    PetscDSSetExactSolution(prob, PRES, lowMach_quadratic_p, NULL) >> checkError;
    PetscDSSetExactSolution(prob, TEMP, lowMach_quadratic_T, NULL) >> checkError;
    PetscDSSetExactSolutionTimeDerivative(prob, VEL, lowMach_quadratic_u_t, NULL) >> checkError;
    PetscDSSetExactSolutionTimeDerivative(prob, PRES, NULL, NULL) >> checkError;
    PetscDSSetExactSolutionTimeDerivative(prob, TEMP, lowMach_quadratic_T_t, NULL) >> checkError;
}

Vec ablate::flow::LowMachFlow::SetupSolve(TS &ts) {
    // Setup the solve with the ts
    TSSetDM(ts, mesh->GetDomain()) >> checkError;

    // finish setup and assign flow field
    LowMachFlow_CompleteProblemSetup(ts, &flowSolution);

    // Name the flow field
    PetscObjectSetName((PetscObject)flowSolution, "Low Mach Numerical Solution") >> checkError;
    VecSetOptionsPrefix(flowSolution, "num_sol_") >> checkError;

    // set the dm on the ts
    TSSetDM(ts, mesh->GetDomain()) >> checkError;

    return flowSolution;
}

REGISTER(ablate::flow::Flow, ablate::flow::LowMachFlow, "low mach flow",
    ARG(std::string, "name", "the name of the flow field"),
    ARG(ablate::mesh::Mesh, "mesh", "the mesh"),
    ARG(std::map<std::string TMP_COMMA std::string>, "arguments", "arguments to be passed to petsc"),
    ARG(ablate::parameters::Parameters, "parameters", "incompressible flow parameters"));
