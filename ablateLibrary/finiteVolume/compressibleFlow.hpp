#ifndef ABLATELIBRARY_COMPRESSIBLEFLOW_H
#define ABLATELIBRARY_COMPRESSIBLEFLOW_H

#include <petsc.h>
#include <eos/transport/transportModel.hpp>
#include <string>
#include "eos/eos.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "domain/domain.hpp"
#include "parameters/parameters.hpp"
#include "finiteVolume/finiteVolume.hpp"

namespace ablate::finiteVolume {
class CompressibleFlow : public FiniteVolume {
   public:
    CompressibleFlow(std::string solverId, std::string region, std::shared_ptr<parameters::Parameters> options, std::shared_ptr<eos::EOS> eos, std::shared_ptr<parameters::Parameters> parameters,
                     std::shared_ptr<eos::transport::TransportModel> transport, std::shared_ptr<fluxCalculator::FluxCalculator> = {},
                     std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization = {}, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions = {},
                     std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions = {});
    ~CompressibleFlow() override = default;
};
}  // namespace ablate::flow

#endif  // ABLATELIBRARY_COMPRESSIBLEFLOW_H
