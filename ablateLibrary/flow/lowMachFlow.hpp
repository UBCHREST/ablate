#ifndef ABLATELIBRARY_LOWMACHFLOW_H
#define ABLATELIBRARY_LOWMACHFLOW_H

#include <petsc.h>
#include <string>
#include "flow.hpp"
#include "mesh/mesh.hpp"
#include "parameters/parameters.hpp"

namespace ablate::flow {
class LowMachFlow : public Flow {
   public:
    LowMachFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<parameters::Parameters> options,
                std::vector<std::shared_ptr<FlowFieldSolution>> initialization, std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions,
                std::vector<std::shared_ptr<FlowFieldSolution>> auxiliaryFields);
    virtual ~LowMachFlow() = default;

    void CompleteProblemSetup(TS ts) override;

};
}  // namespace ablate::flow

#endif  // ABLATELIBRARY_LOWMACHFLOW_H
