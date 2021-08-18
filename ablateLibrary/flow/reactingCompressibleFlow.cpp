#include "reactingCompressibleFlow.hpp"
#include <flow/processes/eulerAdvection.hpp>
#include <flow/processes/eulerDiffusion.hpp>
#include <flow/processes/speciesDiffusion.hpp>
#include <flow/processes/tChemReactions.hpp>
#include <utilities/mpiError.hpp>

ablate::flow::ReactingCompressibleFlow::ReactingCompressibleFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<eos::EOS> eosIn,
                                                                 std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<eos::transport::TransportModel> transport,
                                                                 std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorIn, std::shared_ptr<parameters::Parameters> options,
                                                                 std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization,
                                                                 std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                                 std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions)
    : FVFlow(name, mesh, parameters,
             {{.fieldName = "euler", .fieldPrefix = "euler", .components = 2 + mesh->GetDimensions(), .fieldType = FieldType::FV},
              {
                  .fieldName = "densityYi",
                  .fieldPrefix = "densityYi",
                  .components = (PetscInt)eosIn->GetSpecies().size(),
                  .fieldType = FieldType::FV,
                  .componentNames = eosIn->GetSpecies(),
              },
              {.solutionField = false, .fieldName = "T", .fieldPrefix = "T", .components = 1, .fieldType = FieldType::FV},
              {.solutionField = false, .fieldName = "vel", .fieldPrefix = "vel", .components = mesh->GetDimensions(), .fieldType = FieldType::FV},
              {.solutionField = false, .fieldName = "yi", .fieldPrefix = "yi", .components = (PetscInt)eosIn->GetSpecies().size(), .fieldType = FieldType::FV, .componentNames = eosIn->GetSpecies()}},
             {
                 // create assumed processes for compressible flow
                 std::make_shared<ablate::flow::processes::EulerAdvection>(parameters, eosIn, fluxCalculatorIn),
                 std::make_shared<ablate::flow::processes::EulerDiffusion>(eosIn, transport),
                 std::make_shared<ablate::flow::processes::SpeciesDiffusion>(eosIn, transport),
                 std::make_shared<ablate::flow::processes::TChemReactions>(std::dynamic_pointer_cast<eos::TChem>(eosIn) ? std::dynamic_pointer_cast<eos::TChem>(eosIn)
                                                                                                                        : throw std::invalid_argument("The eos must of type eos::TChem")),
             },
             options, initialization, boundaryConditions, {}, exactSolutions) {}

#include "parser/registrar.hpp"
REGISTER(ablate::flow::Flow, ablate::flow::ReactingCompressibleFlow, "reacting compressible finite volume flow", ARG(std::string, "name", "the name of the flow field"),
         ARG(ablate::mesh::Mesh, "mesh", "the  mesh and discretization"), ARG(ablate::eos::EOS, "eos", "the TChem v1 equation of state used to describe the flow"),
         ARG(ablate::parameters::Parameters, "parameters", "the compressible flow parameters cfl, gamma, etc."),
         OPT(ablate::eos::transport::TransportModel, "transport", "the diffusion transport model"),
         OPT(ablate::flow::fluxCalculator::FluxCalculator, "fluxCalculator", "the flux calculator (defaults to AUSM)"), OPT(ablate::parameters::Parameters, "options", "the options passed to PETSc"),
         OPT(std::vector<mathFunctions::FieldFunction>, "initialization", "the flow field initialization"),
         OPT(std::vector<flow::boundaryConditions::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"),
         OPT(std::vector<mathFunctions::FieldFunction>, "exactSolution", "optional exact solutions that can be used for error calculations"));