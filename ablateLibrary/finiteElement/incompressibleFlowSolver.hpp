#ifndef ABLATELIBRARY_INCOMPRESSIBLEFLOW_H
#define ABLATELIBRARY_INCOMPRESSIBLEFLOW_H

#include <petsc.h>
#include <string>
#include "domain/domain.hpp"
#include "finiteElementSolver.hpp"
#include "finiteVolume/boundaryConditions/boundaryCondition.hpp"
#include "mathFunctions/fieldFunction.hpp"
#include "parameters/parameters.hpp"

namespace ablate::finiteElement {
class IncompressibleFlowSolver : public FiniteElementSolver {
   private:
    const std::shared_ptr<parameters::Parameters> parameters;

   public:
    IncompressibleFlowSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options = {},
                             std::shared_ptr<parameters::Parameters> parameters = {}, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions = {},
                             std::vector<std::shared_ptr<mathFunctions::FieldFunction>> auxiliaryFields = {});

    /** SubDomain Register and Setup **/
    void Setup() override;
    void Initialize() override;

    void CompleteFlowInitialization(DM, Vec) override;

   private:
    inline static std::map<std::string, PetscReal> defaultParameters{{"strouhal", 1.0}, {"reynolds", 1.0}, {"peclet", 1.0}, {"mu", 1.0}, {"k", 1.0}, {"cp", 1.0}};
};
}  // namespace ablate::finiteElement

#endif  // ABLATELIBRARY_LOWMACHFLOW_H
