#ifndef ABLATELIBRARY_FVFLOW_HPP
#define ABLATELIBRARY_FVFLOW_HPP

#include <fvSupport.h>
#include <eos/eos.hpp>
#include <string>
#include <vector>
#include "flow.hpp"

namespace ablate::flow {

// forward declare the FlowProcesses
namespace processes {
class FlowProcess;
}

class FVFlow : public Flow {
   public:
    using RHSArbitraryFunction = PetscErrorCode (*)(DM dm, PetscReal time, Vec locXVec, Vec globFVec, void* ctx);
    using ComputeTimeStepFunction = double (*)(TS ts, Flow&, void* ctx);

   private:
    // hold the update functions for flux and point sources
    std::vector<FVMRHSFluxFunctionDescription> rhsFluxFunctionDescriptions;
    std::vector<FVMRHSPointFunctionDescription> rhsPointFunctionDescriptions;

    // allow the use of any arbitrary rhs functions
    std::vector<std::pair<RHSArbitraryFunction, void*>> rhsArbitraryFunctions;
    // functions to update each aux field
    std::vector<FVAuxFieldUpdateFunction> auxFieldUpdateFunctions;
    std::vector<void*> auxFieldUpdateContexts;

    // functions to update the timestep
    std::vector<std::pair<ComputeTimeStepFunction, void*>> timeStepFunctions;

    // Hold the flow processes.  This is mostly just to hold a pointer to them
    std::vector<std::shared_ptr<processes::FlowProcess>> flowProcesses;

    // static function to update the flowfield
    static void ComputeTimeStep(TS, Flow&);

   public:
    FVFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<parameters::Parameters> parameters, std::vector<FlowFieldDescriptor> fieldDescriptors,
           std::vector<std::shared_ptr<processes::FlowProcess>> flowProcesses, std::shared_ptr<parameters::Parameters> options,
           std::vector<std::shared_ptr<mathFunctions::FieldSolution>> initialization, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
           std::vector<std::shared_ptr<mathFunctions::FieldSolution>> auxiliaryFields, std::vector<std::shared_ptr<mathFunctions::FieldSolution>> exactSolution);
    FVFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<parameters::Parameters> parameters, std::vector<std::shared_ptr<FlowFieldDescriptor>> fieldDescriptors,
           std::vector<std::shared_ptr<processes::FlowProcess>> flowProcesses, std::shared_ptr<parameters::Parameters> options,
           std::vector<std::shared_ptr<mathFunctions::FieldSolution>> initialization, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
           std::vector<std::shared_ptr<mathFunctions::FieldSolution>> auxiliaryFields, std::vector<std::shared_ptr<mathFunctions::FieldSolution>> exactSolution);

    ~FVFlow() override = default;

    void CompleteProblemSetup(TS ts) override;

    /**
     * Function passed into PETSc to compute the FV RHS
     * @param dm
     * @param time
     * @param locXVec
     * @param globFVec
     * @param ctx
     * @return
     */
    static PetscErrorCode FVRHSFunctionLocal(DM dm, PetscReal time, Vec locXVec, Vec globFVec, void* ctx);

    /**
     * Register a FVM rhs source flux function
     * @param function
     * @param context
     * @param field
     * @param inputFields
     * @param auxFields
     */
    void RegisterRHSFunction(FVMRHSFluxFunction function, void* context, std::string field, std::vector<std::string> inputFields, std::vector<std::string> auxFields);

    /**
     * Register a FVM rhs point function
     * @param function
     * @param context
     * @param field
     * @param inputFields
     * @param auxFields
     */
    void RegisterRHSFunction(FVMRHSPointFunction function, void* context, std::vector<std::string> fields, std::vector<std::string> inputFields, std::vector<std::string> auxFields);

    /**
     * Register an arbitrary function.  The user is responsible for all work
     * @param function
     * @param context
     */
    void RegisterRHSFunction(RHSArbitraryFunction function, void* context);

    /**
     * Register a auxFieldUpdate
     * @param function
     * @param context
     * @param field
     * @param inputFields
     * @param auxFields
     */
    void RegisterAuxFieldUpdate(FVAuxFieldUpdateFunction function, void* context, std::string auxField);

    /**
     * Register a dtCalculator
     * @param function
     * @param context
     * @param field
     * @param inputFields
     * @param auxFields
     */
    void RegisterComputeTimeStepFunction(ComputeTimeStepFunction function, void* ctx);
};

}  // namespace ablate::flow
#endif  // ABLATELIBRARY_FVFLOW_HPP
