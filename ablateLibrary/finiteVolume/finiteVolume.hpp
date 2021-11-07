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

class FiniteVolume : public solver::Solver, public solver::RHSFunction {
   public:
    using RHSArbitraryFunction = PetscErrorCode (*)(DM dm, PetscReal time, Vec locXVec, Vec locFVec, void* ctx);
    using ComputeTimeStepFunction = double (*)(TS ts, FiniteVolume&, void* ctx);
    using AuxFieldUpdateFunction = PetscErrorCode (*)(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], const PetscScalar* u, PetscScalar* auxField, void* ctx);

   private:
    /**
     * struct to describe how to compute the aux variable update
     */
    struct AuxFieldUpdateFunctionDescription {
        AuxFieldUpdateFunction function;
        void* context;
        std::vector<PetscInt> inputFields;
        PetscInt auxField;
    };

    // hold the update functions for flux and point sources
    std::vector<FVMRHSFluxFunctionDescription> rhsFluxFunctionDescriptions;
    std::vector<FVMRHSPointFunctionDescription> rhsPointFunctionDescriptions;
    std::vector<AuxFieldUpdateFunctionDescription> auxFieldUpdateFunctionDescriptions;

    // allow the use of any arbitrary rhs functions
    std::vector<std::pair<RHSArbitraryFunction, void*>> rhsArbitraryFunctions;

    // functions to update the timestep
    std::vector<std::pair<ComputeTimeStepFunction, void*>> timeStepFunctions;

    // Hold the flow processes.  This is mostly just to hold a pointer to them
    std::vector<std::shared_ptr<processes::Process>> processes;

    // static function to update the flowfield
    static void ComputeTimeStep(TS, ablate::solver::Solver&);

    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization;
    const std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions;
    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions;

    /**
     * Helper function to march over each cell and update the aux Fields
     * @param flow
     * @param time
     * @param locXVec
     * @param updateFunction
     */
    void UpdateAuxFields(PetscReal time, Vec locXVec, Vec locAuxField);

    /**
     * Computes the flux across each face in th region

     */
    void ComputeFlux(PetscReal time, Vec locXVec, Vec locAuxField, Vec locF);

    /**
     * Inserts the boundary conditions into the locXVec
     * @param time
     * @param locXVec
     * @param locAuxField
     */
    void InsertBoundaryValues(PetscReal time, Vec locXVec, Vec locAuxField);

   protected:
    /**
     * Get the cellIS and range over valid cells in this region
     * @param cellIS
     * @param pStart
     * @param pEnd
     * @param points
     */
    void GetCellRange(IS& cellIS, PetscInt& cStart, PetscInt& cEnd, const PetscInt*& cells);

    /**
     * Get the faceIS and range over valid faces in this region
     * @param cellIS
     * @param pStart
     * @param pEnd
     * @param points
     */
    void GetFaceRange(IS& faceIS, PetscInt& fStart, PetscInt& fEnd, const PetscInt*& faces);

    /**
     * Get the valid range over specified depth
     * @param cellIS
     * @param pStart
     * @param pEnd
     * @param points
     */
    void GetRange(PetscInt depth, IS& pointIS, PetscInt& pStart, PetscInt& pEnd, const PetscInt*& points);

    /**
     * Restores the is and range
     * @param cellIS
     * @param pStart
     * @param pEnd
     * @param points
     */
    void RestoreRange(IS& pointIS, PetscInt& pStart, PetscInt& pEnd, const PetscInt*& points);

   public:
    FiniteVolume(std::string solverId, std::shared_ptr<domain::Region>, std::shared_ptr<parameters::Parameters> options, std::vector<std::shared_ptr<processes::Process>> flowProcesses,
                 std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                 std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolution);

    ~FiniteVolume() override;
    /** SubDomain Register and Setup **/
    void Setup() override;
    void Initialize() override;

    /**
     * Function passed into PETSc to compute the FV RHS
     * @param dm
     * @param time
     * @param locXVec
     * @param globFVec
     * @param ctx
     * @return
     */
    PetscErrorCode ComputeRHSFunction(PetscReal time, Vec locXVec, Vec locFVec) override;

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
    void RegisterAuxFieldUpdate(AuxFieldUpdateFunction function, void* context, std::string auxField, std::vector<std::string> inputFields);

    /**
     * Register a dtCalculator
     * @param function
     * @param context
     * @param field
     * @param inputFields
     * @param auxFields
     */
    void RegisterComputeTimeStepFunction(ComputeTimeStepFunction function, void* ctx);

    /**
     * Function to save the subDomain flowField to a viewer
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    void Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) const override;
};
}  // namespace ablate::finiteVolume

#endif  // ABLATELIBRARY_FINITEVOLUME_HPP
