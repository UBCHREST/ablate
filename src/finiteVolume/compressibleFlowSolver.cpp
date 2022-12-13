#include "compressibleFlowSolver.hpp"
#include <utility>
#include "compressibleFlowFields.hpp"
#include "finiteVolume/processes/evTransport.hpp"
#include "finiteVolume/processes/navierStokesTransport.hpp"
#include "finiteVolume/processes/speciesTransport.hpp"
#include "utilities/vectorUtilities.hpp"

ablate::finiteVolume::CompressibleFlowSolver::CompressibleFlowSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options,
                                                                     const std::shared_ptr<eos::EOS>& eosIn, const std::shared_ptr<parameters::Parameters>& parameters,
                                                                     const std::shared_ptr<eos::transport::TransportModel>& transport,
                                                                     const std::shared_ptr<fluxCalculator::FluxCalculator>& fluxCalculatorIn,
                                                                     std::vector<std::shared_ptr<processes::Process>> additionalProcesses,
                                                                     std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions, bool computePhysicsTimeStep)
    : FiniteVolumeSolver(std::move(solverId), std::move(region), std::move(options),
                         utilities::VectorUtilities::Merge(
                             {
                                 // create assumed processes for compressible flow
                                 std::make_shared<ablate::finiteVolume::processes::NavierStokesTransport>(
                                     parameters, eosIn, fluxCalculatorIn, transport, utilities::VectorUtilities::Find<ablate::finiteVolume::processes::PressureGradientScaling>(additionalProcesses)),
                                 std::make_shared<ablate::finiteVolume::processes::SpeciesTransport>(eosIn, fluxCalculatorIn, transport),
                                 std::make_shared<ablate::finiteVolume::processes::EVTransport>(eosIn, fluxCalculatorIn, transport),
                             },
                             additionalProcesses),
                         std::move(boundaryConditions), computePhysicsTimeStep) {}

ablate::finiteVolume::CompressibleFlowSolver::CompressibleFlowSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options,
                                                                     const std::shared_ptr<eos::EOS>& eosIn, const std::shared_ptr<parameters::Parameters>& parameters,
                                                                     const std::shared_ptr<eos::transport::TransportModel>& transport,
                                                                     const std::shared_ptr<fluxCalculator::FluxCalculator>& fluxCalculatorIn,
                                                                     std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions, bool computePhysicsTimeStep)
    : CompressibleFlowSolver(std::move(solverId), std::move(region), std::move(options), eosIn, parameters, transport, fluxCalculatorIn, {}, std::move(boundaryConditions), computePhysicsTimeStep) {}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::finiteVolume::CompressibleFlowSolver, "compressible finite volume flow", ARG(std::string, "id", "the name of the flow field"),
         OPT(ablate::domain::Region, "region", "the region to apply this solver.  Default is entire domain"), OPT(ablate::parameters::Parameters, "options", "the options passed to PETSc"),
         ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"), OPT(ablate::parameters::Parameters, "parameters", "the parameters used for field values"),
         OPT(ablate::eos::transport::TransportModel, "transport", "the diffusion transport model"),
         OPT(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculator", "the flux calculators (defaults to none)"),
         OPT(std::vector<ablate::finiteVolume::processes::Process>, "additionalProcesses", "any additional processes besides euler/yi/ev transport"),
         OPT(std::vector<ablate::finiteVolume::boundaryConditions::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"),
         OPT(bool, "computePhysicsTimeStep", "determines if a physics based time step is used to control the FVM time stepping (default is false)"));