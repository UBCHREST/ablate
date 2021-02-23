#ifndef ABLATELIBRARY_FLOW_H
#define ABLATELIBRARY_FLOW_H

#include <petsc.h>
#include <memory>
#include <string>
#include "mesh/mesh.hpp"
#include "solve/solvable.hpp"
#include "flowFieldSolution.hpp"
#include "boundaryCondition.hpp"

namespace ablate::flow {
class Flow : public solve::Solvable {
   protected:
    const std::shared_ptr<mesh::Mesh> mesh;
    const std::string name;
    const std::vector<std::shared_ptr<FlowFieldSolution>> initialization;
    const std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions;

    Vec flowSolution;

    Flow(std::shared_ptr<mesh::Mesh> mesh, std::string name, std::map<std::string, std::string> arguments, std::vector<std::shared_ptr<FlowFieldSolution>> initialization,
         std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions);
    virtual ~Flow();

   public:
    Vec GetFlowSolution() { return flowSolution; }

    virtual Vec SetupSolve(TS& timeStepper) = 0;
};
}  // namespace ablate::flow

#endif  // ABLATELIBRARY_FLOW_H
