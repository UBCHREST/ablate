#ifndef ABLATELIBRARY_LOWMACHFLOW_H
#define ABLATELIBRARY_LOWMACHFLOW_H

#include <petsc.h>
#include <string>
#include "mesh/mesh.hpp"
#include "flow.hpp"
#include "parameters/parameters.hpp"
#include "parameters/parameters.hpp"

namespace ablate{
namespace flow {
class LowMachFlow : public Flow {
   public:
    LowMachFlow(std::shared_ptr<mesh::Mesh> mesh, std::string name, std::map<std::string, std::string> arguments, std::shared_ptr<parameters::Parameters> parameters);

    Vec SetupSolve(TS& timeStepper) override;
};
}
}

#endif  // ABLATELIBRARY_LOWMACHFLOW_H
