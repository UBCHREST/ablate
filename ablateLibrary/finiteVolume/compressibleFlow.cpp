#include "compressibleFlow.hpp"
#include <finiteVolume/processes/eulerAdvection.hpp>
#include <finiteVolume/processes/eulerDiffusion.hpp>
#include <finiteVolume/processes/speciesDiffusion.hpp>
#include <utilities/mpiError.hpp>

ablate::finiteVolume::CompressibleFlow::CompressibleFlow(std::string name, std::shared_ptr<parameters::Parameters> options, std::shared_ptr<eos::EOS> eosIn, std::shared_ptr<parameters::Parameters> parameters,
                                                 std::shared_ptr<eos::transport::TransportModel> transport, std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorIn,
                                                  std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization,
                                                 std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                 std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions)
    : FiniteVolume(name, options,
             {{.fieldName = "euler", .fieldPrefix = "euler", .components = 2 + domain::NDIMS},
              {
                  .fieldName = "densityYi",
                  .fieldPrefix = "densityYi",
                  .components = (PetscInt)eosIn->GetSpecies().size(),
                  .componentNames = eosIn->GetSpecies(),
              },
              {.fieldName = "T", .fieldPrefix = "T", .components = 1, .fieldLocation =  domain::FieldLocation::AUX},
              {.fieldName = "vel", .fieldPrefix = "vel", .components = domain::NDIMS,.fieldLocation =  domain::FieldLocation::AUX},
              {.fieldName = "yi", .fieldPrefix = "yi", .components = (PetscInt)eosIn->GetSpecies().size(),.fieldLocation =  domain::FieldLocation::AUX}},
             {
                 // create assumed processes for compressible flow
                 std::make_shared<ablate::finiteVolume::processes::EulerAdvection>(parameters, eosIn, fluxCalculatorIn),
                 std::make_shared<ablate::finiteVolume::processes::EulerDiffusion>(eosIn, transport),
                 std::make_shared<ablate::finiteVolume::processes::SpeciesDiffusion>(eosIn, transport),
             },
             initialization, boundaryConditions, exactSolutions) {}

#include "parser/registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::finiteVolume::CompressibleFlow, "compressible finite volume flow",
         ARG(std::string, "name", "the name of the flow field"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSc"),
         ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"),
         OPT(ablate::parameters::Parameters, "parameters", "the parameters used for field values"),
         OPT(ablate::eos::transport::TransportModel, "transport", "the diffusion transport model"),
         OPT(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculator", "the flux calculators (defaults to AUSM)"),
         OPT(std::vector<mathFunctions::FieldFunction>, "initialization", "the flow field initialization"),
         OPT(std::vector<finiteVolume::boundaryConditions::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"),
         OPT(std::vector<mathFunctions::FieldFunction>, "exactSolution", "optional exact solutions that can be used for error calculations"));