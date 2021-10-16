#ifndef ABLATELIBRARY_LOWMACHFLOW_H
#define ABLATELIBRARY_LOWMACHFLOW_H

#include <petsc.h>
#include <map>
#include <string>
#include "finiteElement.hpp"
#include "domain/domain.hpp"
#include "parameters/parameters.hpp"
#include "finiteVolume/boundaryConditions/boundaryCondition.hpp"

namespace ablate::finiteElement {
class LowMachFlow : public FiniteElement {
   private:
    const std::shared_ptr<parameters::Parameters> parameters;

   public:
    LowMachFlow(std::string name, std::shared_ptr<parameters::Parameters> options = {}, std::shared_ptr<parameters::Parameters> parameters = {},
                std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization = {}, std::vector<std::shared_ptr<finiteVolume::boundaryConditions::BoundaryCondition>> boundaryConditions = {},
                std::vector<std::shared_ptr<mathFunctions::FieldFunction>> auxiliaryFields = {}, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions = {});
    virtual ~LowMachFlow() = default;

    void SetupElementDomain() override;
    void CompleteSetup(TS ts) override;

    void CompleteFlowInitialization(DM, Vec) override;

   private:
    inline static std::map<std::string, PetscReal> defaultParameters{{"strouhal", 1.0},
                                                                     {"reynolds", 1.0},
                                                                     {"froude", 1.0},
                                                                     {"peclet", 1.0},
                                                                     {"heatRelease", 1.0},
                                                                     {"gamma", 1.0},
                                                                     {"pth", 1.0},
                                                                     {"mu", 1.0},
                                                                     {"k", 1.0},
                                                                     {"cp", 1.0},
                                                                     {"beta", 1.0},
                                                                     {"gravityDirection", 0}};
};
}  // namespace ablate::flow

#endif  // ABLATELIBRARY_LOWMACHFLOW_H
