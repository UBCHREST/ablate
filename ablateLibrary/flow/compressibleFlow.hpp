#ifndef ABLATELIBRARY_COMPRESSIBLEFLOW_H
#define ABLATELIBRARY_COMPRESSIBLEFLOW_H

#include <petsc.h>
#include <string>
#include "eos/eos.hpp"
#include "flow/fluxDifferencer/fluxDifferencer.hpp"
#include "fvFlow.hpp"
#include "mesh/mesh.hpp"
#include "parameters/parameters.hpp"

namespace ablate::flow {
class CompressibleFlow : public FVFlow {
   public:
    CompressibleFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<eos::EOS> eos, std::shared_ptr<parameters::Parameters> parameters,
                     std::shared_ptr<fluxDifferencer::FluxDifferencer> = {}, std::shared_ptr<parameters::Parameters> options = {},
                     std::vector<std::shared_ptr<mathFunctions::FieldSolution>> initialization = {}, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions = {},
                     std::vector<std::shared_ptr<mathFunctions::FieldSolution>> exactSolutions = {});
    ~CompressibleFlow() override = default;

};
}  // namespace ablate::flow

#endif  // ABLATELIBRARY_COMPRESSIBLEFLOW_H
