#include "flow.hpp"
#include "../utilities/petscError.hpp"
#include "../utilities/petscOptions.hpp"

ablate::flow::Flow::Flow(std::shared_ptr<mesh::Mesh> mesh, std::string name, std::map<std::string, std::string> arguments):
    mesh(mesh), name(name), flowSolution(nullptr)
{
    // append any prefix values
    utilities::PetscOptions::Set(name, arguments);
}

ablate::flow::Flow::~Flow(){
    if(flowSolution){
        VecDestroy(&flowSolution) >> checkError;
    }
}