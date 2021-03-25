#include "lowMachFlow.hpp"
#include "lowMachFlow.h"
#include "parser/registrar.hpp"
#include "utilities/petscError.hpp"

ablate::flow::LowMachFlow::LowMachFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::map<std::string, std::string> arguments, std::shared_ptr<parameters::Parameters> parameters,
                                       std::vector<std::shared_ptr<FlowFieldSolution>> initialization, std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions,
                                       std::vector<std::shared_ptr<FlowFieldSolution>> auxiliaryFields)
    : Flow(mesh, name, arguments, initialization, boundaryConditions, auxiliaryFields) {
    // Setup the problem
    LowMachFlow_SetupDiscretization(flowData, mesh->GetDomain()) >> checkError;

    // Pack up any of the parameters
    PetscScalar constants[TOTAL_LOW_MACH_FLOW_PARAMETERS];
    parameters->Fill(TOTAL_LOW_MACH_FLOW_PARAMETERS, lowMachFlowParametersTypeNames, constants);

    // Start the problem setup
    LowMachFlow_StartProblemSetup(flowData, TOTAL_LOW_MACH_FLOW_PARAMETERS, constants) >> checkError;

    PetscDS prob;
    DMGetDS(mesh->GetDomain(), &prob) >> checkError;
    if (!auxiliaryFields.empty()) {
        LowMachFlow_EnableAuxFields(flowData);
    }

    // Apply any boundary condition
    CompleteInitialization();
}

void ablate::flow::LowMachFlow::SetupSolve(TS &ts) {
    // finish setup and assign flow field
    LowMachFlow_CompleteProblemSetup(flowData, ts);
    ablate::flow::Flow::SetupSolve(ts);
}

REGISTER(ablate::flow::Flow, ablate::flow::LowMachFlow, "low mach flow", ARG(std::string, "name", "the name of the flow field"), ARG(ablate::mesh::Mesh, "mesh", "the mesh"),
         ARG(std::map<std::string TMP_COMMA std::string>, "arguments", "arguments to be passed to petsc"), ARG(ablate::parameters::Parameters, "parameters", "incompressible flow parameters"),
         ARG(std::vector<flow::FlowFieldSolution>, "initialization", "the exact solution used to initialize the flow field"),
         ARG(std::vector<flow::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"),
         OPT(std::vector<flow::FlowFieldSolution>, "auxFields", "enables and sets the update functions for the auxFields"));