#include "compressibleFlow.hpp"
#include <finiteVolume/processes/eulerAdvection.hpp>
#include <finiteVolume/processes/eulerDiffusion.hpp>
#include <finiteVolume/processes/speciesDiffusion.hpp>
#include <utilities/mpiError.hpp>

ablate::finiteVolume::CompressibleFlow::CompressibleFlow(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options, std::shared_ptr<eos::EOS> eosIn,
                                                         std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<eos::transport::TransportModel> transport,
                                                         std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorIn, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization,
                                                         std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                         std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions)
    : FiniteVolume(solverId, region, options,
                   {{.name = "euler", .prefix = "euler", .components = {"rho", "rhoE", "rhoVel" + domain::FieldDescriptor::DIMENSION}},
                    {.name = "densityYi", .prefix = "densityYi", .components = eosIn->GetSpecies()},
                    {.name = "T", .prefix = "T", .type = domain::FieldType::AUX},
                    {.name = "vel", .prefix = "vel", .components = {"vel" + domain::FieldDescriptor::DIMENSION}, .type = domain::FieldType::AUX},
                    {.name = "yi", .prefix = "yi", .components = eosIn->GetSpecies(), .type = domain::FieldType::AUX}},
                   {
                       // create assumed processes for compressible flow
                       std::make_shared<ablate::finiteVolume::processes::EulerAdvection>(parameters, eosIn, fluxCalculatorIn),
                       std::make_shared<ablate::finiteVolume::processes::EulerDiffusion>(eosIn, transport),
                       std::make_shared<ablate::finiteVolume::processes::SpeciesDiffusion>(eosIn, transport),
                   },
                   initialization, boundaryConditions, exactSolutions) {}

#include "parser/registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::finiteVolume::CompressibleFlow, "compressible finite volume flow", ARG(std::string, "id", "the name of the flow field"),
         OPT(domain::Region, "region", "the region to apply this solver.  Default is entire domain"), OPT(ablate::parameters::Parameters, "options", "the options passed to PETSc"),
         ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"), OPT(ablate::parameters::Parameters, "parameters", "the parameters used for field values"),
         OPT(ablate::eos::transport::TransportModel, "transport", "the diffusion transport model"),
         OPT(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculator", "the flux calculators (defaults to AUSM)"),
         OPT(std::vector<mathFunctions::FieldFunction>, "initialization", "the flow field initialization"),
         OPT(std::vector<finiteVolume::boundaryConditions::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"),
         OPT(std::vector<mathFunctions::FieldFunction>, "exactSolution", "optional exact solutions that can be used for error calculations"));