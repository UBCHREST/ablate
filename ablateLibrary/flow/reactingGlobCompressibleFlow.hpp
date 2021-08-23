#ifndef ABLATECLIENTTEMPLATE_REACTINGGLOBCOMPRESSIBLEFLOW_HPP
#define ABLATECLIENTTEMPLATE_REACTINGGLOBCOMPRESSIBLEFLOW_HPP

#include <petsc.h>
#include <eos/transport/transportModel.hpp>
#include <string>
#include "eos/tChem.hpp"
#include "flow/fluxCalculator/fluxCalculator.hpp"
#include "fvFlow.hpp"
#include "mesh/mesh.hpp"
#include "parameters/parameters.hpp"

namespace ablate::flow {
class ReactingGlobCompressibleFlow : public FVFlow {
   public:
    ReactingGlobCompressibleFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<eos::EOS> eos, std::shared_ptr<parameters::Parameters> parameters,
                             std::shared_ptr<eos::transport::TransportModel> transport = {}, std::shared_ptr<fluxCalculator::FluxCalculator> = {}, std::shared_ptr<parameters::Parameters> options = {},
                             std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization = {},
                             std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions = {},
                             std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions = {});
    ~ReactingGlobCompressibleFlow() override = default;
};
}  // namespace ablate::flow

#endif  // ABLATECLIENTTEMPLATE_REACTINGCOMPRESSIBLEFLOW_HPP
