#ifndef ABLATELIBRARY_FINITEVOLUME_HPP
#define ABLATELIBRARY_FINITEVOLUME_HPP

#include <solver/timeStepper.hpp>
#include <string>
#include <vector>
#include "boundaryConditions/boundaryCondition.hpp"
#include "eos/eos.hpp"
#include "fvSupport.h"
#include "mathFunctions/fieldFunction.hpp"
#include "solver/solver.hpp"

namespace ablate::finiteVolume {

// forward declare the FlowProcesses
namespace processes {
class Process;
}

class FiniteVolume : public solver::Solver {
   public:
    using RHSArbitraryFunction = PetscErrorCode (*)(DM dm, PetscReal time, Vec locXVec, Vec globFVec, void* ctx);
    using ComputeTimeStepFunction = double (*)(TS ts, FiniteVolume&, void* ctx);

   private:
    // hold the update functions for flux and point sources
    std::vector<FVMRHSFluxFunctionDescription> rhsFluxFunctionDescriptions;
    std::vector<FVMRHSPointFunctionDescription> rhsPointFunctionDescriptions;
    std::vector<FVAuxFieldUpdateFunctionDescription> auxFieldUpdateFunctionDescriptions;

    // allow the use of any arbitrary rhs functions
    std::vector<std::pair<RHSArbitraryFunction, void*>> rhsArbitraryFunctions;

    // functions to update the timestep
    std::vector<std::pair<ComputeTimeStepFunction, void*>> timeStepFunctions;

    // Hold the flow processes.  This is mostly just to hold a pointer to them
    std::vector<std::shared_ptr<processes::Process>> processes;

    // static function to update the flowfield
    static void ComputeTimeStep(TS, FiniteVolume&);

    // helper function to register fv field
    void RegisterFiniteVolumeField(const domain::FieldDescriptor&);

    const std::vector<domain::FieldDescriptor> fieldDescriptors;
    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization;
    const std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions;
    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions;

   public:
    FiniteVolume(std::string name, std::shared_ptr<parameters::Parameters> options,
                 std::vector<domain::FieldDescriptor> fieldDescriptors,
                 std::vector<std::shared_ptr<processes::Process>> flowProcesses,
                 std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                 std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolution);

    FiniteVolume(std::string name, std::shared_ptr<parameters::Parameters> options,
                 std::vector<std::shared_ptr<domain::FieldDescriptor>> fieldDescriptors,
                 std::vector<std::shared_ptr<processes::Process>> flowProcesses,
                 std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                 std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolution);


    void SetupDomain(std::shared_ptr<ablate::domain::SubDomain> subDomain) override;

    void CompleteSetup(TS ts) override;

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
    void RegisterAuxFieldUpdate(FVAuxFieldUpdateFunction function, void* context, std::string auxField, std::vector<std::string> inputFields);

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
}

#endif  // ABLATELIBRARY_FINITEVOLUME_HPP
