#ifndef ABLATELIBRARY_FLOW_H
#define ABLATELIBRARY_FLOW_H

#include <petsc.h>
#include <memory>
#include <string>
#include "../mesh/mesh.hpp"


namespace ablate{
namespace flow {
class Flow {
   protected:
    const std::shared_ptr<mesh::Mesh> mesh;
    const std::string name;
    Vec flowSolution;

    Flow(std::shared_ptr<mesh::Mesh> mesh, std::string name, std::map<std::string, std::string> arguments);
    virtual ~Flow();
};
}
}

#endif  // ABLATELIBRARY_FLOW_H
