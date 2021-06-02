#ifndef ABLATELIBRARY_LOWMACHFLOW_H
#define ABLATELIBRARY_LOWMACHFLOW_H

#include <petsc.h>
#include <string>
#include "flow.hpp"
#include "mesh/mesh.hpp"
#include "parameters/parameters.hpp"
#include <map>

namespace ablate::flow {
class LowMachFlow : public Flow {
   public:
    LowMachFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<parameters::Parameters> options = {},
                std::vector<std::shared_ptr<FlowFieldSolution>> initialization = {}, std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions = {},
                std::vector<std::shared_ptr<FlowFieldSolution>> auxiliaryFields = {});
    virtual ~LowMachFlow() = default;

    void CompleteProblemSetup(TS ts) override;
    void CompleteFlowInitialization(DM, Vec) override;

   private:
    inline static std::map<std::string, PetscReal> defaultParameters{
        {"strouhal", 1.0},
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
        {"gravityDirection", 0}
    };
};
}  // namespace ablate::flow

#endif  // ABLATELIBRARY_LOWMACHFLOW_H
