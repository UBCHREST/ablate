#ifndef ABLATELIBRARY_INCOMPRESSIBLEFLOW_H
#define ABLATELIBRARY_INCOMPRESSIBLEFLOW_H

#include <petsc.h>
#include <string>
#include "flow.hpp"
#include "mesh/mesh.hpp"
#include "parameters/parameters.hpp"

namespace ablate::flow {
class IncompressibleFlow : public Flow {
   public:
    IncompressibleFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<parameters::Parameters> options = {},
                       std::vector<std::shared_ptr<mathFunctions::FieldSolution>> initialization = {}, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions = {},
                       std::vector<std::shared_ptr<mathFunctions::FieldSolution>> auxiliaryFields = {},std::vector<std::shared_ptr<mathFunctions::FieldSolution>> exactSolutions = {});

    void CompleteProblemSetup(TS ts) override;
    void CompleteFlowInitialization(DM, Vec) override;

   private:
    inline static std::map<std::string, PetscReal> defaultParameters{
        {"strouhal", 1.0},
        {"reynolds", 1.0},
        {"peclet", 1.0},
        {"mu", 1.0},
        {"k", 1.0},
        {"cp", 1.0}
    };
};
}  // namespace ablate::flow

#endif  // ABLATELIBRARY_LOWMACHFLOW_H
