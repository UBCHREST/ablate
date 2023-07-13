#ifndef ABLATELIBRARY_ONEDIMENSIONHEATTRANSFER_HPP
#define ABLATELIBRARY_ONEDIMENSIONHEATTRANSFER_HPP

#include <memory>
#include "solver/cellSolver.hpp"
#include "solver/timeStepper.hpp"

namespace ablate::boundarySolver::physics::subModels {

class OneDimensionHeatTransfer {
   private:
    // Define an enum for the properties needed for the solver
    typedef enum { specificHeat, conductivity, density, total } ConductionProperties;

    // store the solve_id, so that we can save/restore
    const std::string solverId;

    // Hold the original dm for the subModel
    DM subModelDm{};

    // Hold the TS for stepping in time
    TS subModelTs{};

    // The options used to create the dm, ts, ect
    PetscOptions options{};

    // Hold the properties constants
    PetscScalar properties[total]{};

    // Store the maximum surface temperature
    PetscScalar maximumSurfaceTemperature;

    // hold the surfaceCoordinate, this is where we will compute the surface information
    static constexpr PetscScalar surfaceCoordinate[3] = {0.0, 0.0, 0.0};

    // hold the cell of interest or surface cell
    PetscInt surfaceCell = PETSC_DECIDE;

    // hold the boundary vertex or surface vertex
    PetscInt surfaceVertex = PETSC_DECIDE;

    // Hold onto an aux vector to allow easy updating of the heatFlux
    DM auxDm{};

    // Hold onto an aux vector to allow easy updating of the heatFlux
    Vec localAuxVector{};

    //! Store the marker value for the left wall boundary id
    static constexpr PetscInt leftWallId = 1;

    //! Store the marker value for the right wall boundary id
    static constexpr PetscInt rightWallId = 2;

    // store the initialization as it is also used for the far field boundary condition
    const std::shared_ptr<ablate::mathFunctions::MathFunction> initialization;
    /**
     * Setup the discretization on the active dm
     * @param activeDm the active dm
     * @param bcType the kind of boundary condition to add
     * @return
     */
    PetscErrorCode SetupDiscretization(DM activeDm, DMBoundaryConditionType bcType = DM_BC_NATURAL);

    /**
     * Function to update the boundary condition
     * @param ts
     * @return
     */
    static PetscErrorCode UpdateBoundaryCondition(TS ts);

    /**
     * Compute the surface information at this point in time
     * @param dm
     * @param locVec
     * @param surface
     * @return
     */
    PetscErrorCode ComputeSurfaceInformation(DM dm, Vec locVec, PetscReal &surfaceTemperature, PetscReal &heatFlux) const;

   public:
    /**
     * Create a single 1D solid model
     * @param properties the heat transfer properties
     * @param initialization, math function to initialize the temperature
     * @param options
     */
    explicit OneDimensionHeatTransfer(std::string solverId, const std::shared_ptr<ablate::parameters::Parameters> &properties,
                                      const std::shared_ptr<ablate::mathFunctions::MathFunction> &initialization, const std::shared_ptr<ablate::parameters::Parameters> &options = {},
                                      PetscScalar maxSurfaceTemperature = PETSC_DEFAULT);

    /**
     * Clean up the petsc objects
     */
    ~OneDimensionHeatTransfer();

    /**
     * Advances the solver for this cell in time and returns the computed surface state
     * @param dt
     * @return
     */
    PetscErrorCode Solve(PetscReal heatFluxToSurface, PetscReal dt, PetscReal &surfaceTemperature, PetscReal &heatFlux);

    /**
     * Return the sub model TS
     * @return
     */
    [[nodiscard]] TS GetTS() const { return subModelTs; }

    /**
     * Save the state to the PetscViewer
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    PetscErrorCode Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time);

    /**
     * Restore the state from the PetscViewer
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    PetscErrorCode Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time);

   private:
    /**
     * Compute the jacobian term g0 - integrand for the test and basis function term i
     */
    static void JacobianG0Term(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                               const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]);

    /**
     * Compute the jacobian term g3 - integrand for the test function gradient and basis function gradient term
     */
    static void JacobianG3Term(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                               const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]);

    /**
     * Compute the test function integrated.  Note there is only a single field.
     */
    static void WIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[],
                                       const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t,
                                       const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]);

    /**
     * Compute the test function integrated.  Note there is only a single field.
     */
    static void WIntegrandTestGradientFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[],
                                               const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                               PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]);

    /**
     * Boundary condition when the sublimation/max temperature is applied the surface or the far field
     * @return
     */
    static PetscErrorCode EssentialCoupledWallBC(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx);

    /**
     * Boundary condition during heating
     * @return
     */
    static void NaturalCoupledWallBC(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[],
                                     const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]);

    /**
     * Inline helper function to get the current surface heatFlux
     * @return
     */
    inline PetscErrorCode GetSurfaceHeatFlux(PetscScalar &heatFluxToSurface) const {
        PetscFunctionBegin;
        // Update the heat flux in the auxVector
        const PetscScalar *locAuxArray;
        PetscCall(VecGetArrayRead(localAuxVector, &locAuxArray));

        const PetscScalar *locAuxValue;
        PetscCall(DMPlexPointLocalRead(auxDm, surfaceVertex, locAuxArray, &locAuxValue));

        // Set the heat flux
        heatFluxToSurface = locAuxValue[0];

        // cleanup
        PetscCall(VecRestoreArrayRead(localAuxVector, &locAuxArray));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    /**
     * Inline helper function to set the current surface heatFlux
     * @return
     */
    [[nodiscard]] inline PetscErrorCode SetSurfaceHeatFlux(PetscScalar heatFluxToSurface) const {
        PetscFunctionBegin;
        // Update the heat flux in the auxVector
        PetscScalar *locAuxArray;
        PetscCall(VecGetArray(localAuxVector, &locAuxArray));

        PetscScalar *locAuxValue;
        PetscCall(DMPlexPointLocalRef(auxDm, surfaceVertex, locAuxArray, &locAuxValue));

        // Set the heat flux
        locAuxValue[0] = heatFluxToSurface;

        // cleanup
        PetscCall(VecRestoreArray(localAuxVector, &locAuxArray));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
};

}  // namespace ablate::boundarySolver::physics::subModels
#endif  // ABLATELIBRARY_SOLIDHEATTRANSFER_HPP
