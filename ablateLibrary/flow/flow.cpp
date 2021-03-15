#include "flow.hpp"
#include "../utilities/petscError.hpp"
#include "../utilities/petscOptions.hpp"
#include "utilities/petscError.hpp"

ablate::flow::Flow::Flow(std::shared_ptr<mesh::Mesh> mesh, std::string name, std::map<std::string, std::string> arguments, std::vector<std::shared_ptr<FlowFieldSolution>> initialization,
                         std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions)
    : mesh(mesh), name(name), initialization(initialization), boundaryConditions(boundaryConditions) {
    // append any prefix values
    utilities::PetscOptions::Set(name, arguments);

    // setup the flow data
    FlowCreate(&flowData) >> checkError;
}

ablate::flow::Flow::~Flow() { FlowDestroy(&flowData) >> checkError; }

std::optional<int> ablate::flow::Flow::GetFieldId(const std::string& fieldName) {
    PetscBool found;
    PetscInt index;

    PetscEListFind(flowData->numberFlowFields, flowData->flowFieldNames, fieldName.c_str(), &index, &found) >> checkError;

    if (found) {
        return index;
    } else {
        return {};
    }
}
