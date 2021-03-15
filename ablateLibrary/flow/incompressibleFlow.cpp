#include "incompressibleFlow.hpp"
#include <stdexcept>
#include "incompressibleFlow.h"
#include "parser/registrar.hpp"
#include "utilities/petscError.hpp"

ablate::flow::IncompressibleFlow::IncompressibleFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::map<std::string, std::string> arguments,
                                                     std::shared_ptr<parameters::Parameters> parameters, std::vector<std::shared_ptr<FlowFieldSolution>> initialization,
                                                     std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions)
    : Flow(mesh, name, arguments, initialization, boundaryConditions) {
    // Setup the problem
    IncompressibleFlow_SetupDiscretization(flowData, mesh->GetDomain()) >> checkError;

    // Pack up any of the parameters
    PetscScalar constants[TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS];
    parameters->Fill(TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS, incompressibleFlowParametersTypeNames, constants);

    // Start the problem setup
    IncompressibleFlow_StartProblemSetup(flowData, TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS, constants) >> checkError;

    // Apply any boundary conditions
    PetscDS prob;
    DMGetDS(mesh->GetDomain(), &prob) >> checkError;

    // add each boundary condition
    for (auto boundary : boundaryConditions) {
        PetscInt id = boundary->GetLabelId();
        auto fieldId = GetFieldId(boundary->GetFieldName());
        if (!fieldId) {
            throw std::invalid_argument("unknown field for boundary: " + boundary->GetFieldName());
        }

        PetscDSAddBoundary(prob,
                           DM_BC_ESSENTIAL,
                           boundary->GetBoundaryName().c_str(),
                           boundary->GetLabelName().c_str(),
                           fieldId.value(),
                           0,
                           NULL,
                           (void (*)(void))boundary->GetBoundaryFunction(),
                           (void (*)(void))boundary->GetBoundaryTimeDerivativeFunction(),
                           1,
                           &id,
                           boundary->GetContext()) >>
            checkError;
    }

    // Set the exact solution
    for (auto exact : initialization) {
        auto fieldId = GetFieldId(exact->GetName());
        if (!fieldId) {
            throw std::invalid_argument("unknown field for initialization: " + exact->GetName());
        }

        PetscDSSetExactSolution(prob, fieldId.value(), exact->GetSolutionField().GetPetscFunction(), exact->GetSolutionField().GetContext()) >> checkError;
        PetscDSSetExactSolutionTimeDerivative(prob, fieldId.value(), exact->GetTimeDerivative().GetPetscFunction(), exact->GetTimeDerivative().GetContext()) >> checkError;
    }
}

void ablate::flow::IncompressibleFlow::SetupSolve(TS &ts) {
    // Setup the solve with the ts
    TSSetDM(ts, mesh->GetDomain()) >> checkError;

    // finish setup and assign flow field
    IncompressibleFlow_CompleteProblemSetup(flowData, ts);

    // Initialize the flow field
    DMComputeExactSolution(mesh->GetDomain(), 0, flowData->flowField, NULL) >> checkError;

    // Name the flow field
    PetscObjectSetName((PetscObject)(flowData->flowField), "Incompressible Flow Numerical Solution") >> checkError;
    VecSetOptionsPrefix(flowData->flowField, "num_sol_") >> checkError;

    // set the dm on the ts
    TSSetDM(ts, mesh->GetDomain()) >> checkError;
}

REGISTER(ablate::flow::Flow, ablate::flow::IncompressibleFlow, "incompressible flow", ARG(std::string, "name", "the name of the flow field"), ARG(ablate::mesh::Mesh, "mesh", "the mesh"),
         ARG(std::map<std::string TMP_COMMA std::string>, "arguments", "arguments to be passed to petsc"), ARG(ablate::parameters::Parameters, "parameters", "incompressible flow parameters"),
         ARG(std::vector<flow::FlowFieldSolution>, "initialization", "the exact solution used to initialize the flow field"),
         ARG(std::vector<flow::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"));
