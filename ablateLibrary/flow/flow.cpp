#include "flow.hpp"
#include "../utilities/petscError.hpp"

ablate::flow::Flow::Flow(std::shared_ptr<mesh::Mesh> mesh, std::string name, std::map<std::string, std::string> arguments):
    mesh(mesh), name(name), flowSolution(nullptr)
{
}

ablate::flow::Flow::~Flow(){
    if(flowSolution){
        VecDestroy(&flowSolution) >> checkError;
    }
}