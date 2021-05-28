#ifndef ABLATELIBRARY_COMPRESSIBLEFLOW_H
#define ABLATELIBRARY_COMPRESSIBLEFLOW_H

#include <petsc.h>
#include <eos/eos.hpp>
#include <string>
#include "flow.hpp"
#include "mesh/mesh.hpp"
#include "parameters/parameters.hpp"

namespace ablate::flow {
class CompressibleFlow : public Flow {
   private:
    std::shared_ptr<eos::EOS> eos;

   public:
    CompressibleFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::map<std::string, std::string> arguments, std::shared_ptr<parameters::Parameters> parameters,
                     std::vector<std::shared_ptr<FlowFieldSolution>> initialization, std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions, std::shared_ptr<eos::EOS> eos);

    void SetupSolve(TS& timeStepper) override;
};
}  // namespace ablate::flow

#endif  // ABLATELIBRARY_COMPRESSIBLEFLOW_H
