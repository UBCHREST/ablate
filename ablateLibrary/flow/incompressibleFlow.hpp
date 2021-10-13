#ifndef ABLATELIBRARY_INCOMPRESSIBLEFLOW_H
#define ABLATELIBRARY_INCOMPRESSIBLEFLOW_H

#include <petsc.h>
#include <string>
#include "flow.hpp"
#include "domain/domain.hpp"
#include "parameters/parameters.hpp"

namespace ablate::flow {
class IncompressibleFlow : public Flow {
   public:
    IncompressibleFlow(std::string name, std::shared_ptr<domain::Domain> mesh, std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<parameters::Parameters> options = {},
                       std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization = {}, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions = {},
                       std::vector<std::shared_ptr<mathFunctions::FieldFunction>> auxiliaryFields = {}, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions = {});

    void CompleteProblemSetup(TS ts) override;
    void CompleteFlowInitialization(DM, Vec) override;

   private:
    inline static std::map<std::string, PetscReal> defaultParameters{{"strouhal", 1.0}, {"reynolds", 1.0}, {"peclet", 1.0}, {"mu", 1.0}, {"k", 1.0}, {"cp", 1.0}};
};
}  // namespace ablate::flow

#endif  // ABLATELIBRARY_LOWMACHFLOW_H
