#include "reactingCompressibleFlow.hpp"
#include <finiteVolume/processes/eulerAdvection.hpp>
#include <finiteVolume/processes/eulerDiffusion.hpp>
#include <finiteVolume/processes/speciesDiffusion.hpp>
#include <finiteVolume/processes/tChemReactions.hpp>
#include <utilities/mpiError.hpp>

ablate::finiteVolume::ReactingCompressibleFlow::ReactingCompressibleFlow(std::string name, std::shared_ptr<parameters::Parameters> options, std::shared_ptr<eos::EOS> eosIn,
                                                                 std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<eos::transport::TransportModel> transport,
                                                                 std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorIn,
                                                                 std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization,
                                                                 std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                                 std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions)
    : FiniteVolume(name, options,
             {{.fieldName = "euler", .fieldPrefix = "euler", .components = 2 +domain::NDIMS},
              {
                  .fieldName = "densityYi",
                  .fieldPrefix = "densityYi",
                  .components = (PetscInt)eosIn->GetSpecies().size(),
                  .componentNames = eosIn->GetSpecies(),
              },
              {.fieldName = "T", .fieldPrefix = "T", .components = 1, .fieldLocation =  domain::FieldLocation::AUX},
              {.fieldName = "vel", .fieldPrefix = "vel", .components = domain::NDIMS, .fieldLocation =  domain::FieldLocation::AUX},
              {.fieldName = "yi", .fieldPrefix = "yi", .components = (PetscInt)eosIn->GetSpecies().size(), .componentNames = eosIn->GetSpecies(), .fieldLocation =  domain::FieldLocation::AUX}},
             {
                 // create assumed processes for compressible flow
                 std::make_shared<ablate::finiteVolume::processes::EulerAdvection>(parameters, eosIn, fluxCalculatorIn),
                 std::make_shared<ablate::finiteVolume::processes::EulerDiffusion>(eosIn, transport),
                 std::make_shared<ablate::finiteVolume::processes::SpeciesDiffusion>(eosIn, transport),
                 std::make_shared<ablate::finiteVolume::processes::TChemReactions>(std::dynamic_pointer_cast<eos::TChem>(eosIn) ? std::dynamic_pointer_cast<eos::TChem>(eosIn)
                                                                                                                        : throw std::invalid_argument("The eos must of type eos::TChem")),
             },
             initialization, boundaryConditions, exactSolutions) {}

#include "parser/registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::finiteVolume::ReactingCompressibleFlow, "reacting compressible finite volume flow",
         ARG(std::string, "name", "the name of the flow field"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSc"),
         ARG(ablate::eos::EOS, "eos", "the TChem v1 equation of state used to describe the flow"),
         ARG(ablate::parameters::Parameters, "parameters", "the compressible flow parameters cfl, gamma, etc."),
         OPT(ablate::eos::transport::TransportModel, "transport", "the diffusion transport model"),
         OPT(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculator", "the flux calculator (defaults to AUSM)"),
         OPT(std::vector<mathFunctions::FieldFunction>, "initialization", "the flow field initialization"),
         OPT(std::vector<finiteVolume::boundaryConditions::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"),
         OPT(std::vector<mathFunctions::FieldFunction>, "exactSolution", "optional exact solutions that can be used for error calculations"));