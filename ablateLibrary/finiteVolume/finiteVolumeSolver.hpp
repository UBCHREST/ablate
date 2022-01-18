#ifndef ABLATELIBRARY_FINITEVOLUMESOLVER_HPP
#define ABLATELIBRARY_FINITEVOLUMESOLVER_HPP

#include <string>
#include <vector>
#include "boundaryConditions/boundaryCondition.hpp"
#include "eos/eos.hpp"
#include "fvSupport.h"
#include "mathFunctions/fieldFunction.hpp"
#include "solver/cellSolver.hpp"
#include "solver/solver.hpp"
#include "solver/timeStepper.hpp"

namespace ablate::finiteVolume {

// forward declare the FlowProcesses
namespace processes {
class Process;
}

class FiniteVolumeSolver : public solver::CellSolver, public solver::RHSFunction {
   public:
    using RHSArbitraryFunction = PetscErrorCode (*)(const FiniteVolumeSolver&, DM dm, PetscReal time, Vec locXVec, Vec locFVec, void* ctx);
    using ComputeTimeStepFunction = double (*)(TS ts, FiniteVolumeSolver&, void* ctx);
    using FVMRHSFluxFunction = PetscErrorCode (*)(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[], const PetscScalar fieldR[],
                                                  const PetscScalar gradL[], const PetscScalar gradR[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar auxL[],
                                                  const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[], PetscScalar flux[], void* ctx);
    using FVMRHSPointFunction = PetscErrorCode (*)(PetscInt dim, const PetscFVCellGeom* cg, const PetscInt uOff[], const PetscScalar u[], const PetscScalar* const gradU[], const PetscInt aOff[],
                                                   const PetscScalar a[], const PetscScalar* const gradA[], PetscScalar f[], void* ctx);

   private:
    /**
     * struct to describe how to compute RHS finite volume flux source terms
     */
    struct FluxFunctionDescription {
        FVMRHSFluxFunction function;
        void* context;

        PetscInt field;
        std::vector<PetscInt> inputFields;
        std::vector<PetscInt> auxFields;
    };

    /**
     * struct to describe how to compute RHS finite volume point source terms
     */
    struct PointFunctionDescription {
        FVMRHSPointFunction function;
        void* context;

        std::vector<PetscInt> fields;
        std::vector<PetscInt> inputFields;
        std::vector<PetscInt> auxFields;
    };

    // hold the update functions for flux and point sources
    std::vector<FluxFunctionDescription> rhsFluxFunctionDescriptions;
    std::vector<PointFunctionDescription> rhsPointFunctionDescriptions;

    // allow the use of any arbitrary rhs functions
    std::vector<std::pair<RHSArbitraryFunction, void*>> rhsArbitraryFunctions;

    // functions to update the timestep
    std::vector<std::pair<ComputeTimeStepFunction, void*>> timeStepFunctions;

    // Hold the flow processes.  This is mostly just to hold a pointer to them
    std::vector<std::shared_ptr<processes::Process>> processes;

    // static function to update the flowfield
    static void ComputeTimeStep(TS, ablate::solver::Solver&);

    const std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions;

    /**
     * Computes the flux across each face in th region

     */
    void ComputeSourceTerms(PetscReal time, Vec locXVec, Vec locAuxField, Vec locF);

    /**
     * support call to compute the gradients in each cell.  This also limits the gradient based upon
     * the limiter
     */
    void ComputeFieldGradients(const domain::Field& field, Vec xLocalVec, Vec& gradLocVec, DM& dmGrad);

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
    void ComputePointSourceTerms(DM dm, PetscDS ds, PetscInt totDim, const PetscScalar* xArray, DM dmAux, PetscDS dsAux, PetscInt totDimAux, const PetscScalar* auxArray, DM faceDM,
                                 const PetscScalar* faceGeomArray, DM cellDM, const PetscScalar* cellGeomArray, std::vector<DM>& dmGrads, std::vector<const PetscScalar*>& locGradArrays,
                                 std::vector<DM>& dmAuxGrads, std::vector<const PetscScalar*>& locAuxGradArrays, PetscScalar* locFArray);

   public:
    FiniteVolumeSolver(std::string solverId, std::shared_ptr<domain::Region>, std::shared_ptr<parameters::Parameters> options, std::vector<std::shared_ptr<processes::Process>> flowProcesses,
                       std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions);

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
    void RegisterRHSFunction(FVMRHSFluxFunction function, void* context, const std::string& field, const std::vector<std::string>& inputFields, const std::vector<std::string>& auxFields);

    /**
     * Register a FVM rhs point function
     * @param function
     * @param context
     * @param field
     * @param inputFields
     * @param auxFields
     */
    void RegisterRHSFunction(FVMRHSPointFunction function, void* context, const std::vector<std::string>& fields, const std::vector<std::string>& inputFields,
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
    void RegisterComputeTimeStepFunction(ComputeTimeStepFunction function, void* ctx);
};
}  // namespace ablate::finiteVolume

#endif  // ABLATELIBRARY_FINITEVOLUMESOLVER_HPP
