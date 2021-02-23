#include "flow.hpp"
#include "../utilities/petscError.hpp"
#include "../utilities/petscOptions.hpp"

ablate::flow::Flow::Flow(std::shared_ptr<mesh::Mesh> mesh, std::string name, std::map<std::string, std::string> arguments, std::vector<std::shared_ptr<FlowFieldSolution>> initialization, std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions) :mesh(mesh), name(name), flowSolution(nullptr), initialization(initialization), boundaryConditions(boundaryConditions){
    // append any prefix values
    utilities::PetscOptions::Set(name, arguments);
}

ablate::flow::Flow::~Flow() {
    if (flowSolution) {
        VecDestroy(&flowSolution) >> checkError;
    }
}