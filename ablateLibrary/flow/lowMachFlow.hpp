#ifndef ABLATELIBRARY_LOWMACHFLOW_H
#define ABLATELIBRARY_LOWMACHFLOW_H

#include <petsc.h>
#include <string>
#include "flow.hpp"
#include "mesh/mesh.hpp"
#include "parameters/parameters.hpp"

namespace ablate::flow {
class LowMachFlow : public Flow {
   public:
    LowMachFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::map<std::string, std::string> arguments, std::shared_ptr<parameters::Parameters> parameters);

    Vec SetupSolve(TS& timeStepper) override;
};
}  // namespace ablate::flow

#endif  // ABLATELIBRARY_LOWMACHFLOW_H
