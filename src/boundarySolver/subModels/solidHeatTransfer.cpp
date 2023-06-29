#include "solidHeatTransfer.hpp"
#include "utilities/constants.hpp"
#include "domain/RBF/rbfSupport.hpp"

ablate::boundarySolver::subModels::SolidHeatTransfer::SolidHeatTransfer(const std::shared_ptr<ablate::parameters::Parameters> &propertiesIn,
                                                                        const std::shared_ptr<ablate::mathFunctions::MathFunction>& initialization,
                                                                        const std::shared_ptr<ablate::parameters::Parameters> &optionsIn) {
    // Create a petsc options
    PetscOptionsCreate(&options) >> utilities::PetscUtilities::checkError;

    // Fill in the options
    if (optionsIn) {
        optionsIn->Fill(options);
    }

    // Add any required default values if needed
    ablate::utilities::PetscUtilities::Set(options, "-ts_type", "beuler", false);
    ablate::utilities::PetscUtilities::Set(options, "-ts_max_steps", "100000", false);
    ablate::utilities::PetscUtilities::Set(options, "-snes_error_if_not_converged", nullptr, false);
    ablate::utilities::PetscUtilities::Set(options, "-pc_type", "lu", false);
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
    maximumSurfaceTemperature = propertiesIn->Get<PetscScalar>("maximumSurfaceTemperature", utilities::Constants::large);
    farFieldTemperature = propertiesIn->Get<PetscScalar>("farFieldTemperature", 300.0);

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
    TSCreate(PETSC_COMM_SELF, &ts) >> utilities::PetscUtilities::checkError;
    PetscObjectSetOptions((PetscObject)ts, options) >> utilities::PetscUtilities::checkError;
    TSSetDM(ts, subModelDm) >> utilities::PetscUtilities::checkError;
    DMTSSetBoundaryLocal(subModelDm, DMPlexTSComputeBoundary, nullptr) >> utilities::PetscUtilities::checkError;
    DMTSSetIFunctionLocal(subModelDm, DMPlexTSComputeIFunctionFEM, nullptr) >> utilities::PetscUtilities::checkError;
    DMTSSetIJacobianLocal(subModelDm, DMPlexTSComputeIJacobianFEM, nullptr) >> utilities::PetscUtilities::checkError;
    TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP) >> utilities::PetscUtilities::checkError;
    TSSetFromOptions(ts) >> utilities::PetscUtilities::checkError;

    // create the first global vectorr
    Vec u;
    DMCreateGlobalVector(subModelDm, &u) >> utilities::PetscUtilities::checkError;
    PetscObjectSetName((PetscObject)u, "temperature") >> utilities::PetscUtilities::checkError;
    TSSetSolution(ts, u) >> utilities::PetscUtilities::checkError;

    // Set the initial conditions using a math function
    VecZeroEntries(u) >> utilities::PetscUtilities::checkError;

    // Create the array for field function (temperature)
    mathFunctions::PetscFunction functions[1] = {initialization->GetPetscFunction()};
    void *context[1] = {initialization->GetContext()};

    DMProjectFunction(subModelDm, 0.0, functions, context, INSERT_VALUES, u) >> utilities::PetscUtilities::checkError;

    // precompute some of the required information
    DMPlexGetContainingCell(subModelDm, surfaceCoordinate, &surfaceCell) >> utilities::PetscUtilities::checkError;

}

ablate::boundarySolver::subModels::SolidHeatTransfer::~SolidHeatTransfer() {
    if(subModelDm){
        DMDestroy(&subModelDm) >> utilities::PetscUtilities::checkError;
    }
    if(ts){
        TSDestroy(&ts) >> utilities::PetscUtilities::checkError;
    }
    if (options) {
        utilities::PetscUtilities::PetscOptionsDestroyAndCheck("ablate::boundarySolver::subModels::SolidHeatTransfer", &options);
    }
}
PetscErrorCode ablate::boundarySolver::subModels::SolidHeatTransfer::SetupDiscretization(DM activeDm, DMBoundaryConditionType bcType) {
    PetscFunctionBeginUser;

    // make sure that the dimension is set
    PetscInt dim;
    PetscCall(DMGetDimension(activeDm, &dim));

    /* Create finite element */
    PetscFE fe;
    PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, 1, PETSC_TRUE, 1 /*degree  of space */, PETSC_DETERMINE, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, "temperature"));

    /* Set discretization and boundary conditions for each mesh */
    PetscCall(DMSetField(activeDm, 0, nullptr, (PetscObject)fe));
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
    const PetscInt leftWallId = 1;
    switch (bcType) {
        case DM_BC_ESSENTIAL:
            std::cout << "switching bc to DM_BC_ESSENTIAL" << std::endl;
            PetscCall(PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "coupledWall", label, 1, &leftWallId, 0, 0, nullptr, (void (*)())EssentialCoupledWallBC, nullptr, &maximumSurfaceTemperature, nullptr));
            break;
        case DM_BC_NATURAL:
            std::cout << "switching bc to DM_BC_NATURAL" << std::endl;
            PetscInt coupledWallId;
            PetscCall(PetscDSAddBoundary(ds, DM_BC_NATURAL, "coupledWall", label, 1, &leftWallId, 0, 0, nullptr, nullptr, nullptr, nullptr, &coupledWallId));
            PetscWeakForm wf;
            PetscCall(PetscDSGetBoundary(ds, coupledWallId, &wf, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
            PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, leftWallId, 0, 0, 0, NaturalCoupledWallBC, 0, nullptr));

            break;
        default:
            throw std::invalid_argument("Unable to handle BC type");
    }

    // Add the far field BC
    const PetscInt rightWallId = 2;
    PetscCall(PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "farFieldWall", label, 1, &rightWallId, 0, 0, nullptr, (void (*)(void))EssentialCoupledWallBC, nullptr, &farFieldTemperature, nullptr));

    // Set the constants for the properties
    PetscCall(PetscDSSetConstants(ds, total, properties));

    // copy over the discratation
    PetscCall(PetscFEDestroy(&fe));

    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * Compute the jacobian term g0 - integrand for the test and basis function term i
 */
void ablate::boundarySolver::subModels::SolidHeatTransfer::JacobianG0Term(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[],
                                                                          const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[],
                                                                          const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants,
                                                                          const PetscScalar constants[], PetscScalar g3[]) {
    for (PetscInt d = 0; d < dim; ++d) {
        g3[d * dim + d] = u_tShift * constants[density] * constants[specificHeat];
    }
}

/**
 * Compute the jacobian term g3 - integrand for the test function gradient and basis function gradient term
 */
void ablate::boundarySolver::subModels::SolidHeatTransfer::JacobianG3Term(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[],
                                                                          const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[],
                                                                          const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants,
                                                                          const PetscScalar constants[], PetscScalar g3[]) {
    for (PetscInt d = 0; d < dim; ++d) {
        g3[d * dim + d] = constants[conductivity];
    }
}

/**
 * Compute the test function integrated.  Note there is only a single field.
 */
void ablate::boundarySolver::subModels::SolidHeatTransfer::WIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[],
                                                                                  const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[],
                                                                                  const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[],
                                                                                  PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]) {
    f0[0] = constants[density] * constants[specificHeat] * u_t[0];
}

/**
 * Compute the test function integrated.  Note there is only a single field.
 */
void ablate::boundarySolver::subModels::SolidHeatTransfer::WIntegrandTestGradientFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[],
                                                                                          const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[],
                                                                                          const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t,
                                                                                          const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]) {
    for (PetscInt d = 0; d < dim; ++d) {
        f1[d] = constants[conductivity] * u_x[d];
    }
}

PetscErrorCode ablate::boundarySolver::subModels::SolidHeatTransfer::EssentialCoupledWallBC(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nc, PetscScalar *u, void *ctx) {
    *u = *((PetscScalar *)ctx);
    return PETSC_SUCCESS;
}

void ablate::boundarySolver::subModels::SolidHeatTransfer::NaturalCoupledWallBC(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[],
                                 const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]) {
    // The normal is facing out, so scale the heat flux by -1
    f0[0] = -10000.0;
}
