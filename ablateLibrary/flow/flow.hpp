#ifndef ABLATELIBRARY_FLOW_HPP
#define ABLATELIBRARY_FLOW_HPP

#include <petsc.h>
#include <memory>
#include <optional>
#include <parameters/parameters.hpp>
#include <string>
#include "boundaryCondition.hpp"
#include "flowFieldDescriptor.hpp"
#include "flowFieldSolution.hpp"
#include "mesh/mesh.hpp"
#include "solve/solvable.hpp"
namespace ablate::flow {

class Flow : public solve::Solvable {
   private:
    // descriptions to the fields on the dm
    std::vector<FlowFieldDescriptor> flowFieldDescriptors;

    // descriptions to the fields on the auxDM
    std::vector<FlowFieldDescriptor> auxFieldDescriptors;

    static PetscErrorCode TSPreStepFunction(TS ts);
    static PetscErrorCode TSPostStepFunction(TS ts);

   protected:
    const std::string name;

    // holds non solution vector fields
    DM dm;
    DM auxDM;

    // The solution to the flow
    Vec flowField;

    // The aux field to the flow
    Vec auxField;

    // pre and post step functions for the flow
    std::vector<std::function<void(TS ts, const Flow&)>> preStepFunctions;
    std::vector<std::function<void(TS ts, const Flow&)>> postStepFunctions;

    const std::vector<std::shared_ptr<FlowFieldSolution>> initialization;
    const std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions;
    const std::vector<std::shared_ptr<FlowFieldSolution>> auxiliaryFields;

    // Register the field
    void RegisterField(FlowFieldDescriptor flowFieldDescription);
    void RegisterAuxField(FlowFieldDescriptor flowFieldDescription);
    void FinalizeRegisterFields();

    // Quick reference to used properties,
    PetscInt dim;

   public:
    Flow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<parameters::Parameters> options, std::vector<std::shared_ptr<FlowFieldSolution>> initialization,
         std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions, std::vector<std::shared_ptr<FlowFieldSolution>> auxiliaryFields );
    virtual ~Flow();

    virtual void CompleteProblemSetup(TS ts);

    const std::string& GetName() const { return name; }

    void SetupSolve(TS& timeStepper) override{
        CompleteProblemSetup(timeStepper);
    }

    Vec GetSolutionVector() override { return flowField; }

    std::optional<int> GetFieldId(const std::string& fieldName);

    std::optional<int> GetAuxFieldId(const std::string& fieldName);
};
}  // namespace ablate::flow

#endif  // ABLATELIBRARY_FLOW_H
