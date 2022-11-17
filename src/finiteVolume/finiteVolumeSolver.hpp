#ifndef ABLATELIBRARY_FINITEVOLUMESOLVER_HPP
#define ABLATELIBRARY_FINITEVOLUMESOLVER_HPP

#include <string>
#include <vector>
#include "boundaryConditions/boundaryCondition.hpp"
#include "cellInterpolant.hpp"
#include "eos/eos.hpp"
#include "faceInterpolant.hpp"
#include "mathFunctions/fieldFunction.hpp"
#include "solver/cellSolver.hpp"
#include "solver/solver.hpp"
#include "solver/timeStepper.hpp"
#include "utilities/vectorUtilities.hpp"

namespace ablate::finiteVolume {

// forward declare the FlowProcesses
namespace processes {
class Process;
}

class FiniteVolumeSolver : public solver::CellSolver, public solver::RHSFunction, public io::Serializable, public solver::BoundaryFunction {
   public:
    using RHSArbitraryFunction = PetscErrorCode (*)(const FiniteVolumeSolver&, DM dm, PetscReal time, Vec locXVec, Vec locFVec, void* ctx);
    using ComputeTimeStepFunction = double (*)(TS ts, FiniteVolumeSolver&, void* ctx);

   private:
    /**
     * struct to describe the compute timestamp functions
     */
    struct ComputeTimeStepDescription {
        ComputeTimeStepFunction function;
        void* context;
        std::string name; /**used for output**/
    };

    // hold the update functions for flux and point sources
    std::vector<CellInterpolant::DiscontinuousFluxFunctionDescription> discontinuousFluxFunctionDescriptions;
    std::vector<FaceInterpolant::ContinuousFluxFunctionDescription> continuousFluxFunctionDescriptions;
    std::vector<CellInterpolant::PointFunctionDescription> pointFunctionDescriptions;

    // allow the use of any arbitrary rhs functions
    std::vector<std::pair<RHSArbitraryFunction, void*>> rhsArbitraryFunctions;

    // functions to update the timestep
    const bool computePhysicsTimeStep;
    std::vector<ComputeTimeStepDescription> timeStepFunctions;

    // Hold the flow processes.  This is mostly just to hold a pointer to them
    std::vector<std::shared_ptr<processes::Process>> processes;

    // static function to update the flowfield
    static void EnforceTimeStep(TS ts, ablate::solver::Solver& solver);

    // store the boundary conditions
    const std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions;

    //! hold the class responsible for compute face values;
    std::unique_ptr<FaceInterpolant> faceInterpolant = nullptr;

    //! hold the class responsible for compute cell based values;
    std::unique_ptr<CellInterpolant> cellInterpolant = nullptr;

    //!! Store an region of all cells not in the ghost for faster iteration
    std::shared_ptr<domain::Region> solverRegionMinusGhost;

   public:
    FiniteVolumeSolver(std::string solverId, std::shared_ptr<domain::Region>, std::shared_ptr<parameters::Parameters> options, std::vector<std::shared_ptr<processes::Process>> flowProcesses,
                       std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions, bool computePhysicsTimeStep = false);

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
     * Updates any traditional ghost node boundary
     * @param time
     * @param locX
     * @param locX_t
     * @return
     */
    PetscErrorCode ComputeBoundary(PetscReal time, Vec locX, Vec locX_t) override;

    /**
     * Register a FVM rhs discontinuous flux function
     * @param function
     * @param context
     * @param field
     * @param inputFields
     * @param auxFields
     */
    void RegisterRHSFunction(CellInterpolant::DiscontinuousFluxFunction function, void* context, const std::string& field, const std::vector<std::string>& inputFields,
                             const std::vector<std::string>& auxFields);

    /**
     * Register a FVM rhs continuous flux function
     * @param function
     * @param context
     * @param field
     * @param inputFields
     * @param auxFields
     */
    void RegisterRHSFunction(FaceInterpolant::ContinuousFluxFunction function, void* context, const std::string& field, const std::vector<std::string>& inputFields,
                             const std::vector<std::string>& auxFields);

    /**
     * Register a FVM rhs point function
     * @param function
     * @param context
     * @param field
     * @param inputFields
     * @param auxFields
     */
    void RegisterRHSFunction(CellInterpolant::PointFunction function, void* context, const std::vector<std::string>& fields, const std::vector<std::string>& inputFields,
                             const std::vector<std::string>& auxFields);

    /**
     * Register an arbitrary function.  The user is responsible for all work
     * @param function
     * @param context
     */
    void RegisterRHSFunction(RHSArbitraryFunction function, void* context);

    /**
     * Register a dtCalculator
     * @param function
     * @param context
     * @param field
     * @param inputFields
     * @param auxFields
     */
    void RegisterComputeTimeStepFunction(ComputeTimeStepFunction function, void* ctx, std::string name);

    /**
     * Computes the individual time steps useful for output/debugging.  This does not enforce the time step
     */
    std::map<std::string, double> ComputePhysicsTimeSteps(TS);

    /**
     * Returns true if any of the processes are marked as serializable
     * @return
     */
    bool Serialize() const override;

    /**
     * only required function, returns the id of the object.  Should be unique for the simulation
     * @return
     */
    const std::string& GetId() const override { return GetSolverId(); }

    /**
     * Save the state to the PetscViewer
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    void Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;

    /**
     * Restore the state from the PetscViewer
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    void Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;

    /**
     * Get the cellIS and range over valid cells in this region without ghost cells (boundary or mpi)
     * @param cellIS
     * @param pStart
     * @param pEnd
     * @param points
     */
    void GetCellRangeWithoutGhost(solver::Range& faceRange) const;

    /**
     * Returns first instance of process of type specifed
     * @tparam T
     * @return
     */
    template <class T>
    std::shared_ptr<T> FindProcess() {
        return utilities::VectorUtilities::Find<T>(processes);
    }

    /**
     * Called to update the aux variables
     * @param time
     * @param locX
     * @return
     */
    PetscErrorCode PreRHSFunction(PetscReal time, Vec locX) override;
};
}  // namespace ablate::finiteVolume

#endif  // ABLATELIBRARY_FINITEVOLUMESOLVER_HPP
