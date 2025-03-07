#include "compressibleFlowSolver.hpp"
#include <utility>
#include "compressibleFlowFields.hpp"
#include "finiteVolume/processes/compactCompressibleNSSpeciesSingleProgressTransport.hpp"
#include "finiteVolume/processes/compactCompressibleNSSpeciesTransport.hpp"
#include "finiteVolume/processes/evTransport.hpp"
#include "finiteVolume/processes/navierStokesTransport.hpp"
#include "finiteVolume/processes/speciesTransport.hpp"
#include "utilities/vectorUtilities.hpp"

/**
 * The compact argument in this constructor is used to define whether we use seperate processes for the Euler (NS) transport, species transport, and EV transport.
 * a value of 0, is all seperate
 * a value of 1, is combined Euler and species transport, with seperate EV transport
 * a value of 2, is combined Euler and species and The single field Progress EV transport (soot number density in this case)
 */
ablate::finiteVolume::CompressibleFlowSolver::CompressibleFlowSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options,
                                                                     const std::shared_ptr<eos::EOS>& eosIn, const std::shared_ptr<parameters::Parameters>& parameters,
                                                                     const std::shared_ptr<eos::transport::TransportModel>& transport,
                                                                     const std::shared_ptr<fluxCalculator::FluxCalculator>& fluxCalculatorIn,
                                                                     std::vector<std::shared_ptr<processes::Process>> additionalProcesses,
                                                                     std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                                     const std::shared_ptr<eos::transport::TransportModel>& evTransport, int compact)
    : FiniteVolumeSolver(
          std::move(solverId), std::move(region), std::move(options),
          compact ? (compact == 1 ? utilities::VectorUtilities::Merge({std::make_shared<ablate::finiteVolume::processes::CompactCompressibleNSSpeciesTransport>(
                                                                           parameters, eosIn, fluxCalculatorIn, transport,
                                                                           utilities::VectorUtilities::Find<ablate::finiteVolume::processes::PressureGradientScaling>(additionalProcesses)),
                                                                       std::make_shared<ablate::finiteVolume::processes::EVTransport>(eosIn, fluxCalculatorIn, evTransport ? evTransport : transport)},
                                                                      additionalProcesses)
                                  : utilities::VectorUtilities::Merge({std::make_shared<ablate::finiteVolume::processes::CompactCompressibleNSSpeciesSingleProgressTransport>(
                                                                          parameters, eosIn, fluxCalculatorIn, transport, evTransport ? evTransport : transport,
                                                                          utilities::VectorUtilities::Find<ablate::finiteVolume::processes::PressureGradientScaling>(additionalProcesses))},
                                                                      additionalProcesses))
                  : utilities::VectorUtilities::Merge(
                        {
                            // create assumed processes for compressible flow
                            std::make_shared<ablate::finiteVolume::processes::NavierStokesTransport>(
                                parameters, eosIn, fluxCalculatorIn, transport, utilities::VectorUtilities::Find<ablate::finiteVolume::processes::PressureGradientScaling>(additionalProcesses)),
                            std::make_shared<ablate::finiteVolume::processes::SpeciesTransport>(eosIn, fluxCalculatorIn, transport, parameters),
                            std::make_shared<ablate::finiteVolume::processes::EVTransport>(eosIn, fluxCalculatorIn, evTransport ? evTransport : transport),
                        },
                        additionalProcesses),
          std::move(boundaryConditions)) {}

ablate::finiteVolume::CompressibleFlowSolver::CompressibleFlowSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options,
                                                                     const std::shared_ptr<eos::EOS>& eosIn, const std::shared_ptr<parameters::Parameters>& parameters,
                                                                     const std::shared_ptr<eos::transport::TransportModel>& transport,
                                                                     const std::shared_ptr<fluxCalculator::FluxCalculator>& fluxCalculatorIn,
                                                                     std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                                     const std::shared_ptr<eos::transport::TransportModel>& evTransport, int compact)
    : CompressibleFlowSolver(std::move(solverId), std::move(region), std::move(options), eosIn, parameters, transport, fluxCalculatorIn, {}, std::move(boundaryConditions), evTransport, compact) {}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::finiteVolume::CompressibleFlowSolver, "compressible finite volume flow", ARG(std::string, "id", "the name of the flow field"),
         OPT(ablate::domain::Region, "region", "the region to apply this solver.  Default is entire domain"), OPT(ablate::parameters::Parameters, "options", "the options passed to PETSc"),
         ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"), OPT(ablate::parameters::Parameters, "parameters", "the parameters used for field values"),
         OPT(ablate::eos::transport::TransportModel, "transport", "the diffusion transport model"),
         OPT(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculator", "the flux calculators (defaults to none)"),
         OPT(std::vector<ablate::finiteVolume::processes::Process>, "additionalProcesses", "any additional processes besides euler/yi/ev transport"),
         OPT(std::vector<ablate::finiteVolume::boundaryConditions::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"),
         OPT(ablate::eos::transport::TransportModel, "evTransport", "when provided, this model will be used for ev transport instead of default"),
         OPT(int, "compact", "Integer value describing whether to treat all the transport seperately, partially combined, or fully combined (see commented code above constructor for values)"));