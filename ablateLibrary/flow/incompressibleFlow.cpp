#include "incompressibleFlow.hpp"
#include <stdexcept>
#include "incompressibleFlow.h"
#include "parser/registrar.hpp"
#include "utilities/petscError.hpp"

ablate::flow::IncompressibleFlow::IncompressibleFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::map<std::string, std::string> arguments,
                                                     std::shared_ptr<parameters::Parameters> parameters, std::vector<std::shared_ptr<FlowFieldSolution>> initialization,
                                                     std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions, std::vector<std::shared_ptr<FlowFieldSolution>> auxiliaryFields)
    : Flow(mesh, name, arguments, initialization, boundaryConditions, auxiliaryFields) {
    // Setup the problem
    IncompressibleFlow_SetupDiscretization(flowData, mesh->GetDomain()) >> checkError;

    // Pack up any of the parameters
    PetscScalar constants[TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS];
    parameters->Fill(TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS, incompressibleFlowParametersTypeNames, constants);

    // Start the problem setup
    IncompressibleFlow_StartProblemSetup(flowData, TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS, constants) >> checkError;

    if (!auxiliaryFields.empty()) {
        IncompressibleFlow_EnableAuxFields(flowData);
    }

    CompleteInitialization();
}

void ablate::flow::IncompressibleFlow::SetupSolve(TS &ts) {
    // finish setup and assign flow field
    IncompressibleFlow_CompleteProblemSetup(flowData, ts) >> checkError;
    ablate::flow::Flow::SetupSolve(ts);
}

REGISTER(ablate::flow::Flow, ablate::flow::IncompressibleFlow, "incompressible flow", ARG(std::string, "name", "the name of the flow field"), ARG(ablate::mesh::Mesh, "mesh", "the mesh"),
         ARG(std::map<std::string TMP_COMMA std::string>, "arguments", "arguments to be passed to petsc"), ARG(ablate::parameters::Parameters, "parameters", "incompressible flow parameters"),
         ARG(std::vector<flow::FlowFieldSolution>, "initialization", "the exact solution used to initialize the flow field"),
         ARG(std::vector<flow::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"),
         OPT(std::vector<flow::FlowFieldSolution>, "auxFields", "enables and sets the update functions for the auxFields"));
