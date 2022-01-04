#ifndef ABLATECLIBRARYE_REACTINGCOMPRESSIBLEFLOW_HPP
#define ABLATECLIBRARYE_REACTINGCOMPRESSIBLEFLOW_HPP

#include <petsc.h>
#include <string>
#include "compressibleFlowSolver.hpp"
#include "domain/domain.hpp"
#include "eos/tChem.hpp"
#include "eos/transport/transportModel.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "parameters/parameters.hpp"

namespace ablate::finiteVolume {
class ReactingCompressibleFlowSolver : public CompressibleFlowSolver {
   public:
    ReactingCompressibleFlowSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options, std::shared_ptr<eos::EOS> eos,
                                   std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<eos::transport::TransportModel> transport = {},
                                   std::shared_ptr<fluxCalculator::FluxCalculator> = {}, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions = {},
                                   std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions = {});
    ~ReactingCompressibleFlowSolver() override = default;
};
}  // namespace ablate::finiteVolume

#endif  // ABLATECLIENTTEMPLATE_REACTINGCOMPRESSIBLEFLOW_HPP
