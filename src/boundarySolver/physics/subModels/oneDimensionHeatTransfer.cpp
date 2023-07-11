#include "oneDimensionHeatTransfer.hpp"
#include <petsc/private/petscfeimpl.h>

#include <utility>
#include "domain/RBF/rbfSupport.hpp"
#include "utilities/constants.hpp"

ablate::boundarySolver::physics::subModels::OneDimensionHeatTransfer::OneDimensionHeatTransfer(std::string solverIdIn, const std::shared_ptr<ablate::parameters::Parameters> &propertiesIn,
                                                                                               const std::shared_ptr<ablate::mathFunctions::MathFunction> &initializationIn,
                                                                                               const std::shared_ptr<ablate::parameters::Parameters> &optionsIn, PetscScalar maxSurfaceTemperature)
    : solverId(std::move(solverIdIn)), initialization(initializationIn) {
    // Create a petsc options
    PetscOptionsCreate(&options) >> utilities::PetscUtilities::checkError;

    // Fill in the options
    if (optionsIn) {
        optionsIn->Fill(options);
    }

    // Add any required default values if needed
    ablate::utilities::PetscUtilities::Set(options, "-ts_type", "beuler", false);
    ablate::utilities::PetscUtilities::Set(options, "-ts_max_steps", "10000000", false);
    ablate::utilities::PetscUtilities::Set(options, "-ts_adapt_type", "basic", false);
    ablate::utilities::PetscUtilities::Set(options, "-snes_error_if_not_converged", nullptr, false);
    ablate::utilities::PetscUtilities::Set(options, "-pc_type", "lu", false);
    ablate::utilities::PetscUtilities::Set(options, "-ts_adapt_monitor", "", false);
    // Set the mesh parameters
    ablate::utilities::PetscUtilities::Set(options, "-dm_plex_separate_marker", nullptr, false);
    ablate::utilities::PetscUtilities::Set(options, "-dm_plex_dim", "1", false);
    ablate::utilities::PetscUtilities::Set(options, "-dm_plex_box_faces", "15", false);
    ablate::utilities::PetscUtilities::Set(options, "-dm_plex_box_upper", "0.1", false);

    // Get the properties
    properties[specificHeat] = propertiesIn->GetExpect<PetscScalar>("specificHeat");
    properties[conductivity] = propertiesIn->GetExpect<PetscScalar>("conductivity");
    properties[density] = propertiesIn->GetExpect<PetscScalar>("density");

    // Get the local surface information
    maximumSurfaceTemperature = maxSurfaceTemperature;

    // Create the mesh
    DMCreate(PETSC_COMM_SELF, &subModelDm) >> utilities::PetscUtilities::checkError;
    DMSetType(subModelDm, DMPLEX) >> utilities::PetscUtilities::checkError;
    PetscObjectSetOptions((PetscObject)subModelDm, options) >> utilities::PetscUtilities::checkError;
    PetscObjectSetName((PetscObject)subModelDm, "oneDimMesh") >> utilities::PetscUtilities::checkError;
    DMSetFromOptions(subModelDm) >> utilities::PetscUtilities::checkError;
    DMViewFromOptions(subModelDm, nullptr, "-dm_view") >> utilities::PetscUtilities::checkError;

    // Setup the SetupDiscretization on the first dm
    SetupDiscretization(subModelDm) >> utilities::PetscUtilities::checkError;

    // Create the time stepping
    TSCreate(PETSC_COMM_SELF, &subModelTs) >> utilities::PetscUtilities::checkError;
    PetscObjectSetOptions((PetscObject)subModelTs, options) >> utilities::PetscUtilities::checkError;
    TSSetDM(subModelTs, subModelDm) >> utilities::PetscUtilities::checkError;
    DMTSSetBoundaryLocal(subModelDm, DMPlexTSComputeBoundary, nullptr) >> utilities::PetscUtilities::checkError;
    DMTSSetIFunctionLocal(subModelDm, DMPlexTSComputeIFunctionFEM, nullptr) >> utilities::PetscUtilities::checkError;
    DMTSSetIJacobianLocal(subModelDm, DMPlexTSComputeIJacobianFEM, nullptr) >> utilities::PetscUtilities::checkError;
    TSSetExactFinalTime(subModelTs, TS_EXACTFINALTIME_MATCHSTEP) >> utilities::PetscUtilities::checkError;

    // Only bother to update the boundary condition if there is a max surface temperature
    if (maxSurfaceTemperature >= 0) {
        TSSetPreStep(subModelTs, UpdateBoundaryCondition) >> utilities::PetscUtilities::checkError;
    }

    TSSetFromOptions(subModelTs) >> utilities::PetscUtilities::checkError;
    TSSetApplicationContext(subModelTs, this) >> utilities::PetscUtilities::checkError;

    // create the first global vector
    Vec u;
    DMCreateGlobalVector(subModelDm, &u) >> utilities::PetscUtilities::checkError;
    PetscObjectSetName((PetscObject)u, solverId.c_str()) >> utilities::PetscUtilities::checkError;
    TSSetSolution(subModelTs, u) >> utilities::PetscUtilities::checkError;

    // Set the initial conditions using a math function
    VecZeroEntries(u) >> utilities::PetscUtilities::checkError;

    // Create the array for field function (temperature)
    mathFunctions::PetscFunction functions[1] = {initialization->GetPetscFunction()};
    void *context[1] = {initialization->GetContext()};

    DMProjectFunction(subModelDm, 0.0, functions, context, INSERT_VALUES, u) >> utilities::PetscUtilities::checkError;

    // precompute some of the required information
    DMPlexGetContainingCell(subModelDm, surfaceCoordinate, &surfaceCell) >> utilities::PetscUtilities::checkError;

    // determine the point that we need to apply a boundary condition
    {
        DMLabel label;
        DMGetLabel(subModelDm, "marker", &label) >> utilities::PetscUtilities::checkError;

        // Get the label of points in this
        PetscInt pStart, pEnd;
        DMPlexGetChart(subModelDm, &pStart, &pEnd) >> utilities::PetscUtilities::checkError;

        // get the global section
        PetscSection section;
        DMGetSection(subModelDm, &section) >> utilities::PetscUtilities::checkError;

        // Determine the BC Node
        for (PetscInt p = pStart; p < pEnd; ++p) {
            // Get the dof here
            PetscInt dof;
            PetscSectionGetDof(section, p, &dof) >> utilities::PetscUtilities::checkError;
            // Get the label here
            PetscInt bcValue;
            DMLabelGetValue(label, p, &bcValue) >> utilities::PetscUtilities::checkError;

            if (dof && bcValue == leftWallId) {
                if (surfaceVertex == PETSC_DECIDE) {
                    surfaceVertex = p;
                } else {
                    throw std::invalid_argument("Multiple boundary nodes have been located.");
                }
            }
        }
    }

    // Create an auxDm and aux vector containing
    DMClone(subModelDm, &auxDm);

    // Set a single fe field for heatFlux
    PetscFE feHeatFlux;
    PetscFECreateLagrange(PETSC_COMM_SELF, 1, 1, PETSC_TRUE, 1 /*degree  of space */, PETSC_DETERMINE, &feHeatFlux) >> utilities::PetscUtilities::checkError;
    PetscObjectSetName((PetscObject)feHeatFlux, "heatFlux") >> utilities::PetscUtilities::checkError;

    /* Set discretization and boundary conditions for each mesh */
    DMSetField(auxDm, 0, nullptr, (PetscObject)feHeatFlux) >> utilities::PetscUtilities::checkError;
    DMCreateDS(auxDm) >> utilities::PetscUtilities::checkError;

    // Give the aux vector to the subModelDm
    DMCreateLocalVector(auxDm, &localAuxVector) >> utilities::PetscUtilities::checkError;
    VecZeroEntries(localAuxVector) >> utilities::PetscUtilities::checkError;
    DMSetAuxiliaryVec(subModelDm, nullptr, 0, 0, localAuxVector) >> utilities::PetscUtilities::checkError;
}

ablate::boundarySolver::physics::subModels::OneDimensionHeatTransfer::~OneDimensionHeatTransfer() {
    if (subModelDm) {
        DMDestroy(&subModelDm) >> utilities::PetscUtilities::checkError;
    }
    if (subModelTs) {
        TSDestroy(&subModelTs) >> utilities::PetscUtilities::checkError;
    }
    if (options) {
        utilities::PetscUtilities::PetscOptionsDestroyAndCheck("ablate::boundarySolver::subModels::SolidHeatTransfer", &options);
    }
    if (localAuxVector) {
        VecDestroy(&localAuxVector) >> utilities::PetscUtilities::checkError;
    }
}
PetscErrorCode ablate::boundarySolver::physics::subModels::OneDimensionHeatTransfer::SetupDiscretization(DM activeDm, DMBoundaryConditionType bcType) {
    PetscFunctionBeginUser;

    // make sure that the dimension is set
    PetscInt dim;
    PetscCall(DMGetDimension(activeDm, &dim));

    /* Create finite element field for temperature */
    PetscFE feTemperature;
    PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, 1, PETSC_TRUE, 1 /*degree  of space */, PETSC_DETERMINE, &feTemperature));
    PetscCall(PetscObjectSetName((PetscObject)feTemperature, "temperature"));

    /* Set discretization and boundary conditions for each mesh */
    PetscCall(DMSetField(activeDm, 0, nullptr, (PetscObject)feTemperature));
    PetscCall(DMCreateDS(activeDm));

    // setup the problem
    PetscDS ds;
    DMLabel label;
    PetscCall(DMGetLabel(activeDm, "marker", &label));
    PetscCall(DMPlexLabelComplete(activeDm, label));
    PetscCall(DMGetDS(activeDm, &ds));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, JacobianG0Term, nullptr, nullptr, JacobianG3Term));
    PetscCall(PetscDSSetResidual(ds, 0, WIntegrandTestFunction, WIntegrandTestGradientFunction));

    // Add in the boundaries
    switch (bcType) {
        case DM_BC_ESSENTIAL:
            PetscCall(PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "coupledWall", label, 1, &leftWallId, 0, 0, nullptr, (void (*)())EssentialCoupledWallBC, nullptr, &maximumSurfaceTemperature, nullptr));
            break;
        case DM_BC_NATURAL:
            PetscInt coupledWallId;
            PetscCall(PetscDSAddBoundary(ds, DM_BC_NATURAL, "coupledWall", label, 1, &leftWallId, 0, 0, nullptr, nullptr, nullptr, nullptr, &coupledWallId));
            PetscWeakForm wf;
            PetscCall(PetscDSGetBoundary(ds, coupledWallId, &wf, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
            PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, leftWallId, 0, 0, 0, NaturalCoupledWallBC, 0, nullptr));

            break;
        default:
            throw std::invalid_argument("Unable to handle BC type");
    }

    // Add the far field BC using the initialization math function
    PetscCall(
        PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "farFieldWall", label, 1, &rightWallId, 0, 0, nullptr, (void (*)())initialization->GetPetscFunction(), nullptr, initialization->GetContext(), nullptr));

    // Set the constants for the properties
    PetscCall(PetscDSSetConstants(ds, total, properties));

    // clean up the fields
    PetscCall(PetscFEDestroy(&feTemperature));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ablate::boundarySolver::physics::subModels::OneDimensionHeatTransfer::UpdateBoundaryCondition(TS ts) {
    PetscFunctionBeginUser;
    DM dm;
    PetscCall(TSGetDM(ts, &dm));

    // Get the pointer to the solidHeatTransfer object
    OneDimensionHeatTransfer *oneDimensionHeatTransfer;
    PetscCall(TSGetApplicationContext(ts, &oneDimensionHeatTransfer));

    // Get the current global vector
    Vec currentGlobalVec;
    PetscCall(TSGetSolution(ts, &currentGlobalVec));

    // Get the current time
    PetscReal time;
    PetscCall(TSGetTime(ts, &time));

    // Get the local vector and fill in any boundary values
    Vec locVec;
    PetscCall(DMGetLocalVector(dm, &locVec));
    PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, locVec, time, nullptr, nullptr, nullptr));
    PetscCall(DMGlobalToLocal(dm, currentGlobalVec, INSERT_VALUES, locVec));

    // Get the surface temperature and heat flux
    PetscReal surfaceTemperature;
    PetscReal heatFlux;
    PetscCall(oneDimensionHeatTransfer->ComputeSurfaceInformation(dm, locVec, surfaceTemperature, heatFlux));

    // Get the array
    const PetscScalar *locVecArray;
    PetscCall(VecGetArrayRead(locVec, &locVecArray));

    // Now determine what kind of boundary we need
    DMBoundaryConditionType neededBcType = DM_BC_NATURAL;
    if (surfaceTemperature >= oneDimensionHeatTransfer->maximumSurfaceTemperature) {
        neededBcType = DM_BC_ESSENTIAL;
    }

    // Check if the heatflux into the surface is greater than what is being applied
    PetscScalar heatFluxToSurface;
    PetscCall(oneDimensionHeatTransfer->GetSurfaceHeatFlux(heatFluxToSurface));
    if (heatFluxToSurface < heatFlux) {
        neededBcType = DM_BC_NATURAL;
    }

    // Get the ds
    PetscDS ds;
    PetscCall(DMGetDS(dm, &ds));
    DMBoundaryConditionType currentBcType;

    // Get the current bc
    constexpr PetscInt coupledWallId = 0;  // assume the boundary is always zero
    PetscCall(PetscDSGetBoundary(ds, coupledWallId, nullptr, &currentBcType, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
    if (currentBcType != neededBcType) {
        // Clone the DM
        DM newDM;
        PetscCall(DMClone(dm, &newDM));

        // Set the aux vector in the new dm
        PetscCall(DMSetAuxiliaryVec(newDM, nullptr, 0, 0, oneDimensionHeatTransfer->localAuxVector));

        // Setup the new dm
        PetscCall(oneDimensionHeatTransfer->SetupDiscretization(newDM, neededBcType));

        // Reset the TS
        PetscCall(TSReset(ts));

        // Create a new global vector
        Vec newGlobalVector;
        PetscCall(DMCreateGlobalVector(newDM, &newGlobalVector));

        // Copy the name
        const char *name;
        PetscCall(PetscObjectGetName((PetscObject)currentGlobalVec, &name));
        PetscCall(PetscObjectSetName((PetscObject)newGlobalVector, name));

        // Map from the local vector back to the global
        PetscCall(DMLocalToGlobal(newDM, locVec, INSERT_VALUES, newGlobalVector));

        // Set in the TS
        PetscCall(TSSetDM(ts, newDM));
        PetscCall(TSSetSolution(ts, newGlobalVector));
    }

    // Cleanup
    PetscCall(DMRestoreLocalVector(dm, &locVec));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ablate::boundarySolver::physics::subModels::OneDimensionHeatTransfer::ComputeSurfaceInformation(DM dm, Vec locVec, PetscReal &surfaceTemperature, PetscReal &heatFlux) const {
    PetscFunctionBeginUser;

    // Map that coordinate back if needed
    PetscScalar refCoord[3];
    PetscCall(DMPlexCoordinatesToReference(dm, surfaceCell, 1, surfaceCoordinate, refCoord));

    // Build a tabulation to compute the values there
    PetscDS ds;
    PetscFE fe;
    PetscCall(DMGetDS(dm, &ds));
    PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
    PetscInt dim;
    PetscCall(DMGetDimension(dm, &dim));

    // Get the cell geometry
    PetscQuadrature q;
    PetscCall(PetscFEGetQuadrature(fe, &q));
    PetscFEGeom feGeometry;
    PetscCall(PetscFECreateCellGeometry(fe, q, &feGeometry));

    PetscTabulation tab;
    const PetscInt nrepl = 1;    // The number of replicas
    const PetscInt nPoints = 1;  // the number of points
    const PetscInt kOrder = 1;   // The number of derivatives calculated
    PetscCall(PetscFECreateTabulation(fe, nrepl, nPoints, refCoord, kOrder, &tab));

    {  // compute the single point values
        const PetscInt pointInBasis = 0;
        // extract the local cell information
        PetscCall(DMPlexComputeCellGeometryFEM(dm, surfaceCell, q, feGeometry.v, feGeometry.J, feGeometry.invJ, feGeometry.detJ));
        PetscScalar *clPhi = nullptr;
        PetscInt cSize;
        PetscCall(DMPlexVecGetClosure(dm, nullptr, locVec, surfaceCell, &cSize, &clPhi));

        // Get the interpolated value
        surfaceTemperature = 0.0;
        PetscCall(PetscFEInterpolateAtPoints_Static(fe, tab, clPhi, &feGeometry, pointInBasis, &surfaceTemperature));
        PetscScalar temperatureGrad[3] = {0.0, 0.0, 0.0};
        PetscCall(PetscFEFreeInterpolateGradient_Static(fe, tab->T[1], clPhi, dim, feGeometry.invJ, nullptr, pointInBasis, temperatureGrad));

        // get the constants
        PetscInt numConstants;
        const PetscScalar *constants;
        PetscCall(PetscDSGetConstants(ds, &numConstants, &constants));
        heatFlux = -temperatureGrad[0] * constants[conductivity];
    }

    // cleanup
    PetscCall(PetscFEDestroyCellGeometry(fe, &feGeometry));
    PetscCall(PetscTabulationDestroy(&tab));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * Compute the jacobian term g0 - integrand for the test and basis function term i
 */
void ablate::boundarySolver::physics::subModels::OneDimensionHeatTransfer::JacobianG0Term(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[],
                                                                                          const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[],
                                                                                          const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t,
                                                                                          PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[],
                                                                                          PetscScalar g3[]) {
    for (PetscInt d = 0; d < dim; ++d) {
        g3[d * dim + d] = u_tShift * constants[density] * constants[specificHeat];
    }
}

/**
 * Compute the jacobian term g3 - integrand for the test function gradient and basis function gradient term
 */
void ablate::boundarySolver::physics::subModels::OneDimensionHeatTransfer::JacobianG3Term(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[],
                                                                                          const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[],
                                                                                          const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t,
                                                                                          PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[],
                                                                                          PetscScalar g3[]) {
    for (PetscInt d = 0; d < dim; ++d) {
        g3[d * dim + d] = constants[conductivity];
    }
}

/**
 * Compute the test function integrated.  Note there is only a single field.
 */
void ablate::boundarySolver::physics::subModels::OneDimensionHeatTransfer::WIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[],
                                                                                                  const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[],
                                                                                                  const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                                                                  PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[],
                                                                                                  PetscScalar f0[]) {
    f0[0] = constants[density] * constants[specificHeat] * u_t[0];
}

/**
 * Compute the test function integrated.  Note there is only a single field.
 */
void ablate::boundarySolver::physics::subModels::OneDimensionHeatTransfer::WIntegrandTestGradientFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[],
                                                                                                          const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                                                                                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[],
                                                                                                          const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[],
                                                                                                          PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]) {
    for (PetscInt d = 0; d < dim; ++d) {
        f1[d] = constants[conductivity] * u_x[d];
    }
}

PetscErrorCode ablate::boundarySolver::physics::subModels::OneDimensionHeatTransfer::EssentialCoupledWallBC(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nc, PetscScalar *u, void *ctx) {
    *u = *((PetscScalar *)ctx);
    return PETSC_SUCCESS;
}

void ablate::boundarySolver::physics::subModels::OneDimensionHeatTransfer::NaturalCoupledWallBC(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[],
                                                                                                const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[],
                                                                                                const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                                                                PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants,
                                                                                                const PetscScalar constants[], PetscScalar f0[]) {
    // The normal is facing out, so scale the heat flux by -1
    f0[0] = -a[aOff[0]];
}
PetscErrorCode ablate::boundarySolver::physics::subModels::OneDimensionHeatTransfer::Solve(PetscReal heatFluxToSurface, PetscReal dt, PetscReal &surfaceTemperature, PetscReal &heatFlux) {
    PetscFunctionBeginHot;
    // Get the current time
    PetscReal time;
    PetscCall(TSGetTime(subModelTs, &time));

    // Do a soft reset on the ode solver
    PetscCall(TSSetMaxTime(subModelTs, dt + time));

    // Update the heat flux in the auxVector
    PetscCall(SetSurfaceHeatFlux(heatFluxToSurface));

    // Step in time
    PetscCall(TSSolve(subModelTs, nullptr));

    // Get the solution vector from the ts
    Vec globalSolutionVector;
    PetscCall(TSGetSolution(subModelTs, &globalSolutionVector));

    // compute the current surface state
    DM activeDM;
    PetscCall(TSGetDM(subModelTs, &activeDM));
    Vec locVec;
    PetscCall(DMGetLocalVector(activeDM, &locVec));
    PetscCall(DMPlexInsertBoundaryValues(activeDM, PETSC_TRUE, locVec, time, nullptr, nullptr, nullptr));
    PetscCall(DMGlobalToLocal(activeDM, globalSolutionVector, INSERT_VALUES, locVec));
    PetscCall(ComputeSurfaceInformation(activeDM, locVec, surfaceTemperature, heatFlux));
    PetscCall(DMRestoreLocalVector(activeDM, &locVec));

    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode ablate::boundarySolver::physics::subModels::OneDimensionHeatTransfer::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    PetscFunctionBegin;
    // Get the current solution from the TS
    Vec solution;
    PetscCall(TSGetSolution(subModelTs, &solution));
    DM dm;
    PetscCall(TSGetDM(subModelTs, &dm));

    // Set the output sequence
    PetscCall(DMSetOutputSequenceNumber(dm, sequenceNumber, time));

    // Write to the file
    PetscCall(VecView(solution, viewer));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ablate::boundarySolver::physics::subModels::OneDimensionHeatTransfer::Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    PetscFunctionBegin;
    // Get the current solution from the TS
    Vec solution;
    PetscCall(TSGetSolution(subModelTs, &solution));
    DM dm;
    PetscCall(TSGetDM(subModelTs, &dm));

    // Set the output sequence
    PetscCall(DMSetOutputSequenceNumber(dm, sequenceNumber, time));

    // Write to the file
    PetscCall(VecLoad(solution, viewer));

    PetscFunctionReturn(PETSC_SUCCESS);
}
