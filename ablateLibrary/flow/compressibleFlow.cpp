#include "compressibleFlow.hpp"
#include "compressibleFlow.h"
#include "parser/registrar.hpp"
#include "utilities/petscError.hpp"

ablate::flow::CompressibleFlow::CompressibleFlow(std::string name, std::shared_ptr<mesh::Mesh> meshIn, std::map<std::string, std::string> arguments, std::shared_ptr<parameters::Parameters> parameters,
                                                 std::vector<std::shared_ptr<FlowFieldSolution>> initialization, std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions)
    : Flow(meshIn, name, arguments, initialization, boundaryConditions, {}) {
    // Setup the problem
    CompressibleFlow_SetupDiscretization(flowData, &mesh->GetDomain()) >> checkError;

    // Pack up any of the parameters
    PetscScalar constants[TOTAL_COMPRESSIBLE_FLOW_PARAMETERS];
    parameters->Fill(TOTAL_COMPRESSIBLE_FLOW_PARAMETERS, compressibleFlowParametersTypeNames, constants);
    CompressibleFlow_StartProblemSetup(flowData, TOTAL_COMPRESSIBLE_FLOW_PARAMETERS, constants) >> checkError;

    // Apply any boundary condition
    CompleteInitialization();
}

void ablate::flow::CompressibleFlow::SetupSolve(TS &ts) {
    // finish setup and assign flow field
    CompressibleFlow_CompleteProblemSetup(flowData, ts);
    ablate::flow::Flow::SetupSolve(ts);
}

REGISTER(ablate::flow::Flow, ablate::flow::CompressibleFlow, "compressible finite volume flow", ARG(std::string, "name", "the name of the flow field"), ARG(ablate::mesh::Mesh, "mesh", "the mesh"),
         ARG(std::map<std::string TMP_COMMA std::string>, "arguments", "arguments to be passed to petsc"), ARG(ablate::parameters::Parameters, "parameters", "compressible flow parameters"),
         ARG(std::vector<flow::FlowFieldSolution>, "initialization", "the exact solution used to initialize the flow field"),
         OPT(std::vector<flow::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"));