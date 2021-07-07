#include "reactingCompressibleFlow.hpp"
#include <flow/processes/eulerAdvection.hpp>
#include <flow/processes/eulerDiffusion.hpp>
#include <flow/processes/tChemReactions.hpp>
#include <utilities/mpiError.hpp>
#include "compressibleFlow.hpp"

ablate::flow::ReactingCompressibleFlow::ReactingCompressibleFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<eos::TChem> eosIn,
                                                                 std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<fluxDifferencer::FluxDifferencer> fluxDifferencerIn,
                                                                 std::shared_ptr<parameters::Parameters> options, std::vector<std::shared_ptr<mathFunctions::FieldSolution>> initialization,
                                                                 std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                                 std::vector<std::shared_ptr<mathFunctions::FieldSolution>> exactSolutions)
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
              {.solutionField = false, .fieldName = "vel", .fieldPrefix = "vel", .components = mesh->GetDimensions(), .fieldType = FieldType::FV}},
             {
                 // create assumed processes for compressible flow
//                 std::make_shared<ablate::flow::processes::EulerAdvection>(parameters, eosIn, fluxDifferencerIn),
//                 std::make_shared<ablate::flow::processes::EulerDiffusion>(parameters, eosIn),
                 std::make_shared<ablate::flow::processes::TChemReactions>(eosIn),
             },
             options, initialization, boundaryConditions, {}, exactSolutions) {}
