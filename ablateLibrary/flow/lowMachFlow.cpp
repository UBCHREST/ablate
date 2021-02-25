#include "lowMachFlow.hpp"
#include "lowMachFlow.h"
#include "parser/registrar.hpp"
#include "utilities/petscError.hpp"

ablate::flow::LowMachFlow::LowMachFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::map<std::string, std::string> arguments, std::shared_ptr<parameters::Parameters> parameters,
                                       std::vector<std::shared_ptr<FlowFieldSolution>> initialization, std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions)
    : Flow(mesh, name, arguments, initialization, boundaryConditions) {
    // Setup the problem
    LowMachFlow_SetupDiscretization(mesh->GetDomain()) >> checkError;

    // Pack up any of the parameters
    PetscScalar constants[TOTAL_LOW_MACH_FLOW_PARAMETERS];
    parameters->Fill(TOTAL_LOW_MACH_FLOW_PARAMETERS, lowMachFlowParametersTypeNames, constants);

    // Start the problem setup
    LowMachFlow_StartProblemSetup(mesh->GetDomain(), TOTAL_LOW_MACH_FLOW_PARAMETERS, constants) >> checkError;

    // Apply any boundary condition
    PetscDS prob;
    DMGetDS(mesh->GetDomain(), &prob) >> checkError;

    // add each boundary condition
    for (auto boundary : boundaryConditions) {
        PetscInt id = boundary->GetLabelId();
        PetscDSAddBoundary(prob,
                           DM_BC_ESSENTIAL,
                           boundary->GetBoundaryName().c_str(),
                           boundary->GetLabelName().c_str(),
                           GetFieldId(boundary->GetFieldName()),
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

        PetscDSSetExactSolution(prob, fieldId, exact->GetSolutionField().GetPetscFunction(), exact->GetSolutionField().GetContext()) >> checkError;
        PetscDSSetExactSolutionTimeDerivative(prob, fieldId, exact->GetTimeDerivative().GetPetscFunction(), exact->GetTimeDerivative().GetContext()) >> checkError;
    }
}

void ablate::flow::LowMachFlow::SetupSolve(TS &ts) {
    // Setup the solve with the ts
    TSSetDM(ts, mesh->GetDomain()) >> checkError;

    // finish setup and assign flow field
    LowMachFlow_CompleteProblemSetup(ts, &flowSolution);

    // Name the flow field
    PetscObjectSetName((PetscObject)flowSolution, "Low Mach Numerical Solution") >> checkError;
    VecSetOptionsPrefix(flowSolution, "num_sol_") >> checkError;

    // set the dm on the ts
    TSSetDM(ts, mesh->GetDomain()) >> checkError;
}

int ablate::flow::LowMachFlow::GetFieldId(const std::string &field) {
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

REGISTER(ablate::flow::Flow, ablate::flow::LowMachFlow, "low mach flow", ARG(std::string, "name", "the name of the flow field"), ARG(ablate::mesh::Mesh, "mesh", "the mesh"),
         ARG(std::map<std::string TMP_COMMA std::string>, "arguments", "arguments to be passed to petsc"), ARG(ablate::parameters::Parameters, "parameters", "incompressible flow parameters"),
         ARG(std::vector<flow::FlowFieldSolution>, "initialization", "the exact solution used to initialize the flow field"),
         ARG(std::vector<flow::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"));