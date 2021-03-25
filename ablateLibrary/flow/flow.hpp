#ifndef ABLATELIBRARY_FLOW_HPP
#define ABLATELIBRARY_FLOW_HPP

#include <flow.h>
#include <petsc.h>
#include <memory>
#include <optional>
#include <string>
#include "boundaryCondition.hpp"
#include "flow.h"
#include "flowFieldSolution.hpp"
#include "mesh/mesh.hpp"
#include "solve/solvable.hpp"

namespace ablate::flow {
class Flow : public solve::Solvable {
   protected:
    const std::shared_ptr<mesh::Mesh> mesh;
    const std::string name;
    const std::vector<std::shared_ptr<FlowFieldSolution>> initialization;
    const std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions;
    const std::vector<std::shared_ptr<FlowFieldSolution>> auxiliaryFields;

    // Store the flow data
    FlowData flowData;

    Flow(std::shared_ptr<mesh::Mesh> mesh, std::string name, std::map<std::string, std::string> arguments, std::vector<std::shared_ptr<FlowFieldSolution>> initialization,
         std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions, std::vector<std::shared_ptr<FlowFieldSolution>> auxiliaryFields);
    virtual ~Flow();

    /**
     * Method should be call at the end of each constructor to setup boundary conditions and aux field
     */
    void CompleteInitialization();

   private:
    std::vector<mathFunctions::PetscFunction> auxiliaryFieldFunctions;
    std::vector<void*> auxiliaryFieldContexts;
    static PetscErrorCode UpdateAuxiliaryFields(TS ts, void* ctx);

   public:
    FlowData GetFlowData() { return flowData; }

    mesh::Mesh& GetMesh() { return *mesh; }

    const std::string& GetName() const { return name; }

    virtual void SetupSolve(TS& timeStepper) override;

    Vec GetSolutionVector() override { return flowData->flowField; }

    std::optional<int> GetFieldId(const std::string& fieldName);

    std::optional<int> GetAuxFieldId(const std::string& fieldName);
};
}  // namespace ablate::flow

#endif  // ABLATELIBRARY_FLOW_H
