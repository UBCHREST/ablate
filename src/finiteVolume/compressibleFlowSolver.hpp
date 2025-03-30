#ifndef ABLATELIBRARY_COMPRESSIBLEFLOW_H
#define ABLATELIBRARY_COMPRESSIBLEFLOW_H

#include <petsc.h>
#include <eos/transport/transportModel.hpp>
#include <string>
#include "domain/domain.hpp"
#include "eos/eos.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "finiteVolume/processes/pressureGradientScaling.hpp"
#include "parameters/parameters.hpp"

namespace ablate::finiteVolume {
class CompressibleFlowSolver : public FiniteVolumeSolver {
   public:
    /**
     * Full feature constructor
     * @param solverId
     * @param region
     * @param options
     * @param eos
     * @param parameters
     * @param transport
     * @param additionalProcesses
     * @param extraVariables
     * @param initialization
     * @param boundaryConditions
     * @param exactSolutions
     */
    CompressibleFlowSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options, const std::shared_ptr<eos::EOS>& eos,
                           const std::shared_ptr<parameters::Parameters>& parameters, const std::shared_ptr<eos::transport::TransportModel>& transport,
                           const std::shared_ptr<fluxCalculator::FluxCalculator>& = {}, std::vector<std::shared_ptr<processes::Process>> additionalProcesses = {},
                           std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions = {}, const std::shared_ptr<eos::transport::TransportModel>& evTransport = {},
                           int compact = 0);

    /**
     * Constructor without ev or additional processes
     * @param solverId
     * @param region
     * @param options
     * @param eos
     * @param parameters
     * @param transport
     * @param initialization
     * @param boundaryConditions
     * @param exactSolutions
     */
    CompressibleFlowSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options, const std::shared_ptr<eos::EOS>& eos,
                           const std::shared_ptr<parameters::Parameters>& parameters, const std::shared_ptr<eos::transport::TransportModel>& transport,
                           const std::shared_ptr<fluxCalculator::FluxCalculator>& = {}, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions = {},
                           const std::shared_ptr<eos::transport::TransportModel>& evTransport = {}, int compact = 0);
    ~CompressibleFlowSolver() override = default;
};
}  // namespace ablate::finiteVolume

#endif  // ABLATELIBRARY_COMPRESSIBLEFLOW_H
