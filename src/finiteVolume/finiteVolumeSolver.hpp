#ifndef ABLATELIBRARY_FINITEVOLUMESOLVER_HPP
#define ABLATELIBRARY_FINITEVOLUMESOLVER_HPP

#include <string>
#include <vector>
#include "boundaryConditions/boundaryCondition.hpp"
#include "eos/eos.hpp"
#include "faceInterpolant.hpp"
#include "mathFunctions/fieldFunction.hpp"
#include "solver/cellSolver.hpp"
#include "solver/solver.hpp"
#include "solver/timeStepper.hpp"

namespace ablate::finiteVolume {

// forward declare the FlowProcesses
namespace processes {
class Process;
}

class FiniteVolumeSolver : public solver::CellSolver, public solver::RHSFunction, public io::Serializable {
   public:
    using RHSArbitraryFunction = PetscErrorCode (*)(const FiniteVolumeSolver&, DM dm, PetscReal time, Vec locXVec, Vec locFVec, void* ctx);
    using ComputeTimeStepFunction = double (*)(TS ts, FiniteVolumeSolver&, void* ctx);
    /**
     * Function assumes that the left/right solution and aux variables are discontinuous across the interface
     */
    using DiscontinuousFluxFunction = PetscErrorCode (*)(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscInt aOff[],
                                                         const PetscScalar auxL[], const PetscScalar auxR[], PetscScalar flux[], void* ctx);

    /**
     * Functions that operates on entire cell value.
     */
    using PointFunction = PetscErrorCode (*)(PetscInt dim, PetscReal time, const PetscFVCellGeom* cg, const PetscInt uOff[], const PetscScalar u[], const PetscInt aOff[], const PetscScalar a[],
                                             PetscScalar f[], void* ctx);

   private:
    /**
     * struct to describe how to compute RHS finite volume flux source terms
     */
    struct DiscontinuousFluxFunctionDescription {
        DiscontinuousFluxFunction function;
        void* context;

        PetscInt field;
        std::vector<PetscInt> inputFields;
        std::vector<PetscInt> auxFields;
    };

    /**
     * struct to describe how to compute RHS finite volume point source terms
     */
    struct PointFunctionDescription {
        PointFunction function;
        void* context;

        std::vector<PetscInt> fields;
        std::vector<PetscInt> inputFields;
        std::vector<PetscInt> auxFields;
    };

    /**
     * struct to describe the compute timestamp functions
     */
    struct ComputeTimeStepDescription {
        ComputeTimeStepFunction function;
        void* context;
        std::string name; /**used for output**/
    };

    // hold the update functions for flux and point sources
    std::vector<DiscontinuousFluxFunctionDescription> discontinuousFluxFunctionDescriptions;
    std::vector<FaceInterpolant::ContinuousFluxFunctionDescription> continuousFluxFunctionDescriptions;
    std::vector<PointFunctionDescription> pointFunctionDescriptions;

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

    //! store the dmGrad, these are specific to this finite volume solver
    std::vector<DM> gradientCellDms;

    //! store the gradient dm for each aux variable
    std::vector<DM> auxGradientCellDms;

    //! hold the class responsible for compute face values;
    std::unique_ptr<FaceInterpolant> faceInterpolant = nullptr;

    /**
     * Computes the flux across each face in th region based upon the cell information
     */
    void ComputeCellSourceTerms(PetscReal time, Vec locXVec, Vec locAuxField, Vec locF);

    /**
     * support call to compute the gradients in each cell.  This also limits the gradient based upon
     * the limiter
     */
    void ComputeFieldGradients(const domain::Field& field, Vec xLocalVec, Vec& gradLocVec, DM& dmGrad);

    /**
     * Computes the dmGrad over this region
     * @param dm
     * @param fvm
     * @param faceGeometry
     * @param cellGeometry
     * @param dmGrad
     * @return
     */
    static PetscErrorCode ComputeGradientFVM(DM dm, DMLabel regionLabel, PetscInt regionValue, PetscFV fvm, Vec faceGeometry, Vec cellGeometry, DM* dmGrad);

    /**
     * support call to project to a single face from a side
     */
    void ProjectToFace(const std::vector<domain::Field>& fields, PetscDS ds, const PetscFVFaceGeom& faceGeom, PetscInt cellId, const PetscFVCellGeom& cellGeom, DM dm, const PetscScalar* xArray,
                       const std::vector<DM>& dmGrads, const std::vector<const PetscScalar*>& gradArrays, PetscScalar* u, PetscScalar* grad, bool projectField = true);

    /**
     * Function to compute the flux source terms
     */
    void ComputeFluxSourceTerms(DM dm, PetscDS ds, PetscInt totDim, const PetscScalar* xArray, DM dmAux, PetscDS dsAux, PetscInt totDimAux, const PetscScalar* auxArray, DM faceDM,
                                const PetscScalar* faceGeomArray, DM cellDM, const PetscScalar* cellGeomArray, std::vector<DM>& dmGrads, std::vector<const PetscScalar*>& locGradArrays,
                                std::vector<DM>& dmAuxGrads, std::vector<const PetscScalar*>& locAuxGradArrays, PetscScalar* locFArray);

    /**
     * Function to compute the point source terms
     */
    void ComputePointSourceTerms(DM dm, PetscDS ds, PetscInt totDim, PetscReal time, const PetscScalar* xArray, DM dmAux, PetscDS dsAux, PetscInt totDimAux, const PetscScalar* auxArray, DM faceDM,
                                 const PetscScalar* faceGeomArray, DM cellDM, const PetscScalar* cellGeomArray, std::vector<DM>& dmGrads, std::vector<const PetscScalar*>& locGradArrays,
                                 std::vector<DM>& dmAuxGrads, std::vector<const PetscScalar*>& locAuxGradArrays, PetscScalar* locFArray);

    /**
     * call to make sure local ghost boundary have valid gradient values
     * @param dm
     * @param auxFvm
     * @param localXVec
     * @param gradLocalVec
     * @return
     */
    PetscErrorCode FillGradientBoundary(DM dm, PetscFV auxFvm, Vec localXVec, Vec gradLocalVec);

   public:
    FiniteVolumeSolver(std::string solverId, std::shared_ptr<domain::Region>, std::shared_ptr<parameters::Parameters> options, std::vector<std::shared_ptr<processes::Process>> flowProcesses,
                       std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions, bool computePhysicsTimeStep = false);
    ~FiniteVolumeSolver() override;

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
     * Register a FVM rhs discontinuous flux function
     * @param function
     * @param context
     * @param field
     * @param inputFields
     * @param auxFields
     */
    void RegisterRHSFunction(DiscontinuousFluxFunction function, void* context, const std::string& field, const std::vector<std::string>& inputFields, const std::vector<std::string>& auxFields);

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
    void RegisterRHSFunction(PointFunction function, void* context, const std::vector<std::string>& fields, const std::vector<std::string>& inputFields, const std::vector<std::string>& auxFields);

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
};
}  // namespace ablate::finiteVolume

#endif  // ABLATELIBRARY_FINITEVOLUMESOLVER_HPP
