#include "compressibleFlow.hpp"
#include <finiteVolume/processes/evTransport.hpp>
#include "finiteVolume/processes/eulerTransport.hpp"
#include "finiteVolume/processes/speciesTransport.hpp"
#include "utilities/vectorUtilities.hpp"

ablate::finiteVolume::CompressibleFlow::CompressibleFlow(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options, std::shared_ptr<eos::EOS> eosIn,
                                                         std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<eos::transport::TransportModel> transport,
                                                         std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorIn, std::vector<std::shared_ptr<processes::Process>> additionalProcesses,
                                                         std::vector<std::string> extraVariables, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization,
                                                         std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                         std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions)
    : FiniteVolume(
          solverId, region, options,
          {{.name = processes::FlowProcess::EULER_FIELD, .prefix = processes::FlowProcess::EULER_FIELD, .components = {"rho", "rhoE", "rhoVel" + domain::FieldDescriptor::DIMENSION}},
           {.name = processes::FlowProcess::DENSITY_YI_FIELD, .prefix = processes::FlowProcess::DENSITY_YI_FIELD, .components = eosIn->GetSpecies()},
           {.name = "T", .prefix = "T", .type = domain::FieldType::AUX},
           {.name = "vel", .prefix = "vel", .components = {"vel" + domain::FieldDescriptor::DIMENSION}, .type = domain::FieldType::AUX},
           {.name = processes::FlowProcess::YI_FIELD, .prefix = processes::FlowProcess::YI_FIELD, .components = eosIn->GetSpecies(), .type = domain::FieldType::AUX},
           {.name = processes::FlowProcess::DENSITY_EV_FIELD, .prefix = processes::FlowProcess::DENSITY_EV_FIELD, .components = extraVariables},
           {.name = processes::FlowProcess::EV_FIELD, .prefix = processes::FlowProcess::EV_FIELD, .components = extraVariables, .type = domain::FieldType::AUX}},
          utilities::VectorUtilities::Merge(
              {
                  // create assumed processes for compressible flow
                  std::make_shared<ablate::finiteVolume::processes::EulerTransport>(parameters, eosIn, fluxCalculatorIn, transport),
                  std::make_shared<ablate::finiteVolume::processes::SpeciesTransport>(eosIn, fluxCalculatorIn, transport),
                  std::make_shared<ablate::finiteVolume::processes::EVTransport>(processes::FlowProcess::DENSITY_EV_FIELD, processes::FlowProcess::EV_FIELD, eosIn, fluxCalculatorIn, transport),
              },
              additionalProcesses),
          initialization, boundaryConditions, exactSolutions) {}

ablate::finiteVolume::CompressibleFlow::CompressibleFlow(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options, std::shared_ptr<eos::EOS> eosIn,
                                                         std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<eos::transport::TransportModel> transport,
                                                         std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorIn, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization,
                                                         std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                         std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions)
    : CompressibleFlow(solverId, region, options, eosIn, parameters, transport, fluxCalculatorIn, {}, {}, initialization, boundaryConditions, exactSolutions) {}

#include "parser/registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::finiteVolume::CompressibleFlow, "compressible finite volume flow", ARG(std::string, "id", "the name of the flow field"),
         OPT(domain::Region, "region", "the region to apply this solver.  Default is entire domain"), OPT(ablate::parameters::Parameters, "options", "the options passed to PETSc"),
         ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"), OPT(ablate::parameters::Parameters, "parameters", "the parameters used for field values"),
         OPT(ablate::eos::transport::TransportModel, "transport", "the diffusion transport model"),
         OPT(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculator", "the flux calculators (defaults to none)"),
         OPT(std::vector<ablate::finiteVolume::processes::Process>, "additionalProcesses", "any additional processes besides euler/yi/ev transport"),
         OPT(std::vector<std::string>, "extraVariables", "any additional conserved extra variables"),

         OPT(std::vector<mathFunctions::FieldFunction>, "initialization", "the flow field initialization"),
         OPT(std::vector<finiteVolume::boundaryConditions::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"),
         OPT(std::vector<mathFunctions::FieldFunction>, "exactSolution", "optional exact solutions that can be used for error calculations"));