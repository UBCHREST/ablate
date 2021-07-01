#ifndef ABLATELIBRARY_FLOW_HPP
#define ABLATELIBRARY_FLOW_HPP

#include <petsc.h>
#include <functional>
#include <memory>
#include <monitors/viewable.hpp>
#include <optional>
#include <parameters/parameters.hpp>
#include <string>
#include <vector>
#include "flow/boundaryConditions/boundaryCondition.hpp"
#include "flowFieldDescriptor.hpp"
#include "mathFunctions/fieldSolution.hpp"
#include "mesh/mesh.hpp"
#include "solve/solvable.hpp"

namespace ablate::flow {

class Flow : public solve::Solvable, public monitors::Viewable {
   protected:
    // descriptions to the fields on the dm
    std::vector<FlowFieldDescriptor> flowFieldDescriptors;

    // descriptions to the fields on the auxDM
    std::vector<FlowFieldDescriptor> auxFieldDescriptors;

    static PetscErrorCode TSPreStepFunction(TS ts);
    static PetscErrorCode TSPostStepFunction(TS ts);
    static PetscErrorCode TSPostEvaluateFunction(TS ts);

    /**
     * Private method to add both flow and aux fields, depending upon what is passed in
     * @param flowFieldDescription
     */
    void RegisterField(FlowFieldDescriptor flowFieldDescription, DM dm);

   protected:
    const std::string name;

    // holds non solution vector fields
    std::shared_ptr<mesh::Mesh> dm;
    DM auxDM;

    // The solution to the flow
    Vec flowField;

    // The aux field to the flow
    Vec auxField;

    // pre and post step functions for the flow
    std::vector<std::function<void(TS ts, Flow&)>> preStepFunctions;
    std::vector<std::function<void(TS ts, Flow&)>> postStepFunctions;
    std::vector<std::function<void(TS ts, Flow&)>> postEvaluateFunctions;

    const std::vector<std::shared_ptr<mathFunctions::FieldSolution>> initialization;
    const std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions;
    const std::vector<std::shared_ptr<mathFunctions::FieldSolution>> auxiliaryFieldsUpdaters;
    const std::vector<std::shared_ptr<mathFunctions::FieldSolution>> exactSolutions;

    // Register the field
    void RegisterField(FlowFieldDescriptor flowFieldDescription);
    void FinalizeRegisterFields();

    // Quick reference to used properties,
    PetscInt dim;

    // Petsc options specific to flow. These may be null by default
    PetscOptions petscOptions;

   public:
    Flow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<parameters::Parameters> options,
         std::vector<std::shared_ptr<mathFunctions::FieldSolution>> initialization, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
         std::vector<std::shared_ptr<mathFunctions::FieldSolution>> auxiliaryFields, std::vector<std::shared_ptr<mathFunctions::FieldSolution>> exactSolution);
    virtual ~Flow();

    virtual void CompleteProblemSetup(TS ts);
    virtual void CompleteFlowInitialization(DM, Vec) = 0;

    /**
     * function to update the aux fields.
     */
    static void UpdateAuxFields(TS ts, Flow& flow);

    /**
     * provide interface for all flow output
     * @param viewer
     * @param steps
     * @param time
     * @param u
     */
    void View(PetscViewer viewer, PetscInt steps, PetscReal time, Vec u) const override;

    /**
     * Adds function to be called before each flow step
     * @param preStep
     */
    void RegisterPreStep(std::function<void(TS ts, Flow&)> preStep) { this->preStepFunctions.push_back(preStep); }

    /**
     * Adds function to be called after each flow step
     * @param preStep
     */
    void RegisterPostStep(std::function<void(TS ts, Flow&)> postStep) { this->postStepFunctions.push_back(postStep); }

    /**
     * Adds function after each evaluated.  This is where the solution can be modified if needed.
     * @param postStep
     */
    void RegisterPostEvaluate(std::function<void(TS ts, Flow&)> postEval) { this->postEvaluateFunctions.push_back(postEval); }

    const std::string& GetName() const override { return name; }

    const DM& GetDM() const { return dm->GetDomain(); }

    const DM& GetAuxDM() const { return auxDM; }

    void SetupSolve(TS& timeStepper) override { CompleteProblemSetup(timeStepper); }

    Vec GetAuxField() { return auxField; }

    Vec GetSolutionVector() override { return flowField; }

    std::optional<int> GetFieldId(const std::string& fieldName) const;

    std::optional<int> GetAuxFieldId(const std::string& fieldName) const;
};
}  // namespace ablate::flow

#endif  // ABLATELIBRARY_FLOW_H
