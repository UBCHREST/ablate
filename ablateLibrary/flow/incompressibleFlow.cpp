#include "incompressibleFlow.hpp"
#include <stdexcept>
#include "incompressibleFlow.h"
#include "parser/registrar.hpp"
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

ablate::flow::IncompressibleFlow::IncompressibleFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::map<std::string, std::string> arguments,
                                                     std::shared_ptr<parameters::Parameters> parameters, std::vector<std::shared_ptr<FlowFieldSolution>> initialization,
                                                     std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions)
    : Flow(mesh, name, arguments, initialization, boundaryConditions) {
    // Setup the problem
    IncompressibleFlow_SetupDiscretization(mesh->GetDomain()) >> checkError;

    // Pack up any of the parameters
    PetscScalar constants[TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS];
    parameters->Fill(TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS, incompressibleFlowParametersTypeNames, constants);

    // Start the problem setup
    IncompressibleFlow_StartProblemSetup(mesh->GetDomain(), TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS, constants) >> checkError;

    // Apply any boundary conditions
    PetscDS prob;
    DMGetDS(mesh->GetDomain(), &prob) >> checkError;

    // add each boundary condition
    for(auto boundary: boundaryConditions){
        PetscInt id = boundary->GetLabelId();
        PetscDSAddBoundary(prob,
                           DM_BC_ESSENTIAL,
                           boundary->GetBoundaryName().c_str(),
                           boundary->GetLabelName().c_str(),
                           GetFieldId(boundary->GetFieldName()), 0, NULL, (void (*)(void))boundary->GetBoundaryFunction(), (void (*)(void))boundary->GetBoundaryTimeDerivativeFunction(), 1, &id, boundary->GetContext()) >> checkError;
    }

    // Set the exact solution
    for (auto exact : initialization) {
        auto fieldId = GetFieldId(exact->GetName());

        PetscDSSetExactSolution(prob, fieldId, exact->GetSolutionField().GetPetscFunction(), exact->GetSolutionField().GetContext()) >> checkError;
        PetscDSSetExactSolutionTimeDerivative(prob, fieldId, exact->GetTimeDerivative().GetPetscFunction(), exact->GetTimeDerivative().GetContext()) >> checkError;
    }
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
int ablate::flow::IncompressibleFlow::GetFieldId(const std::string &field) {
    if (field == "velocity") {
        return VEL;
    } else if (field == "pressure") {
        return PRES;
    } else if (field == "temperature") {
        return TEMP;
    } else {
        throw std::invalid_argument("invalid flow field (" + field + ")");
    }
}

REGISTER(ablate::flow::Flow, ablate::flow::IncompressibleFlow, "incompressible flow", ARG(std::string, "name", "the name of the flow field"), ARG(ablate::mesh::Mesh, "mesh", "the mesh"),
         ARG(std::map<std::string TMP_COMMA std::string>, "arguments", "arguments to be passed to petsc"), ARG(ablate::parameters::Parameters, "parameters", "incompressible flow parameters"),
         ARG(std::vector<flow::FlowFieldSolution>, "initialization", "the exact solution used to initialize the flow field"),
         ARG(std::vector<flow::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"));
