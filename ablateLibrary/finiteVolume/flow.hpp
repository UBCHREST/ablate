#ifndef ABLATELIBRARY_FLOW_HPP
#define ABLATELIBRARY_FLOW_HPP

#include <petsc.h>
#include <functional>
#include <io/serializable.hpp>
#include <memory>
#include <optional>
#include <parameters/parameters.hpp>
#include <string>
#include <vector>
#include "flow/boundaryConditions/boundaryCondition.hpp"
#include "mathFunctions/fieldFunction.hpp"
#include "domain/domain.hpp"
#include "monitors/monitorable.hpp"
#include "domain/fieldDescriptor.hpp"
#include "solver/solvable.hpp"

namespace ablate::flow {

class Flow : public solver::Solvable, public io::Serializable, public monitors::Monitorable {
   protected:
    // descriptions to the fields on the dm
    std::vector<domain::FieldDescriptor> flowFieldDescriptors;

    // descriptions to the fields on the auxDM
    std::vector<domain::FieldDescriptor> auxFieldDescriptors;

    static PetscErrorCode TSPreStageFunction(TS ts, PetscReal stagetime);
    static PetscErrorCode TSPreStepFunction(TS ts);
    static PetscErrorCode TSPostStepFunction(TS ts);
    static PetscErrorCode TSPostEvaluateFunction(TS ts);


    const std::string name;

    // holds non solution vector fields
    std::shared_ptr<domain::Domain> dm;



    // pre and post step functions for the flow
    std::vector<std::function<void(TS ts, Flow&)>> preStepFunctions;
    std::vector<std::function<void(TS ts, Flow&, PetscReal)>> preStageFunctions;
    std::vector<std::function<void(TS ts, Flow&)>> postStepFunctions;
    std::vector<std::function<void(TS ts, Flow&)>> postEvaluateFunctions;

    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization;
    const std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions;
    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> auxiliaryFieldsUpdaters;
    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions;

    // Register the field

    // Quick reference to used properties,
    PetscInt dim;

    // Petsc options specific to flow. These may be null by default
    PetscOptions petscOptions;

   public:
    Flow(std::string name, std::shared_ptr<domain::Domain> mesh, std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<parameters::Parameters> options,
         std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
         std::vector<std::shared_ptr<mathFunctions::FieldFunction>> auxiliaryFields, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolution);
    virtual ~Flow();

    virtual void CompleteProblemSetup(TS ts);
    virtual void CompleteFlowInitialization(DM, Vec){};

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
    void Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) const override;

    /**
     * provide interface for all flow restore
     * @param viewer
     * @param steps
     * @param time
     * @param u
     */
    void Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;

    /**
     * Adds function to be called before each flow step
     * @param preStep
     */
    void RegisterPreStep(std::function<void(TS ts, Flow&)> preStep) { this->preStepFunctions.push_back(preStep); }

    /**
     * Adds function to be called before each flow step
     * @param preStep
     */
    void RegisterPreStage(std::function<void(TS ts, Flow&, PetscReal)> preStage) { this->preStageFunctions.push_back(preStage); }

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

    const std::string& GetId() const override { return name; }

    const DM& GetDM() const { return dm->GetDomain(); }

    const DM& GetAuxDM() const { return auxDM; }

    void SetupSolve(TS& timeStepper) override { CompleteProblemSetup(timeStepper); }

    Vec GetAuxField() { return auxField; }

    Vec GetSolutionVector() override { return flowField; }

    std::optional<int> GetFieldId(const std::string& fieldName) const;

    std::optional<int> GetAuxFieldId(const std::string& fieldName) const;

    const domain::FieldDescriptor& GetFieldDescriptor(const std::string& fieldName) const;

    const domain::FieldDescriptor& GetAuxFieldDescriptor(const std::string& fieldName) const;
};
}  // namespace ablate::flow

#endif  // ABLATELIBRARY_FLOW_H
