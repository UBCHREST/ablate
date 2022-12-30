#include "lowMachFlowSolver.hpp"
#include "lowMachFlow.h"
#include "utilities/petscUtilities.hpp"

ablate::finiteElement::LowMachFlowSolver::LowMachFlowSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options,
                                                            std::shared_ptr<parameters::Parameters> parameters, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                            std::vector<std::shared_ptr<mathFunctions::FieldFunction>> auxiliaryFields)
    : FiniteElementSolver(solverId, region, options, boundaryConditions, auxiliaryFields), parameters(parameters) {}

void ablate::finiteElement::LowMachFlowSolver::Setup() {
    FiniteElementSolver::Setup();

    // Make sure that the fields
    if (VEL != subDomain->GetField("velocity").subId) {
        throw std::invalid_argument("The velocity field subId is expected to be " + std::to_string(VEL) + ", but found to be " + std::to_string(subDomain->GetField("velocity").subId));
    }
    if (PRES != subDomain->GetField("pressure").subId) {
        throw std::invalid_argument("The pressure field subId is expected to be " + std::to_string(PRES) + ", but found to be " + std::to_string(subDomain->GetField("pressure").subId));
    }
    if (TEMP != subDomain->GetField("temperature").subId) {
        throw std::invalid_argument("The temperature field subId is expected to be " + std::to_string(TEMP) + ", but found to be " + std::to_string(subDomain->GetField("temperature").subId));
    }

    {
        PetscObject pressure;
        MatNullSpace nullspacePres;

        DMGetField(subDomain->GetDM(), PRES, NULL, &pressure) >> utilities::PetscUtilities::checkError;
        MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nullspacePres) >> utilities::PetscUtilities::checkError;
        PetscObjectCompose(pressure, "nullspace", (PetscObject)nullspacePres) >> utilities::PetscUtilities::checkError;
        MatNullSpaceDestroy(&nullspacePres) >> utilities::PetscUtilities::checkError;
    }

    PetscDS prob;
    DMGetDS(subDomain->GetDM(), &prob) >> utilities::PetscUtilities::checkError;

    // V, W, Q Test Function
    PetscDSSetResidual(prob, VTEST, LowMachFlow_vIntegrandTestFunction, LowMachFlow_vIntegrandTestGradientFunction) >> utilities::PetscUtilities::checkError;
    PetscDSSetResidual(prob, WTEST, LowMachFlow_wIntegrandTestFunction, LowMachFlow_wIntegrandTestGradientFunction) >> utilities::PetscUtilities::checkError;
    PetscDSSetResidual(prob, QTEST, LowMachFlow_qIntegrandTestFunction, NULL) >> utilities::PetscUtilities::checkError;

    PetscDSSetJacobian(prob, VTEST, VEL, LowMachFlow_g0_vu, LowMachFlow_g1_vu, NULL, LowMachFlow_g3_vu) >> utilities::PetscUtilities::checkError;
    PetscDSSetJacobian(prob, VTEST, PRES, NULL, NULL, LowMachFlow_g2_vp, NULL) >> utilities::PetscUtilities::checkError;
    PetscDSSetJacobian(prob, VTEST, TEMP, LowMachFlow_g0_vT, NULL, NULL, NULL) >> utilities::PetscUtilities::checkError;
    PetscDSSetJacobian(prob, QTEST, VEL, LowMachFlow_g0_qu, LowMachFlow_g1_qu, NULL, NULL) >> utilities::PetscUtilities::checkError;
    PetscDSSetJacobian(prob, QTEST, TEMP, LowMachFlow_g0_qT, LowMachFlow_g1_qT, NULL, NULL) >> utilities::PetscUtilities::checkError;
    PetscDSSetJacobian(prob, WTEST, VEL, LowMachFlow_g0_wu, NULL, NULL, NULL) >> utilities::PetscUtilities::checkError;
    PetscDSSetJacobian(prob, WTEST, TEMP, LowMachFlow_g0_wT, LowMachFlow_g1_wT, NULL, LowMachFlow_g3_wT) >> utilities::PetscUtilities::checkError;

    /* Setup constants */;
    PetscReal parameterArray[TOTAL_LOW_MACH_FLOW_PARAMETERS];
    parameters->Fill(TOTAL_LOW_MACH_FLOW_PARAMETERS, lowMachFlowParametersTypeNames, parameterArray, defaultParameters);
    PetscDSSetConstants(prob, TOTAL_LOW_MACH_FLOW_PARAMETERS, parameterArray) >> utilities::PetscUtilities::checkError;
}
static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx) {
    PetscInt d;
    for (d = 0; d < Nc; ++d) u[d] = 0.0;
    return 0;
}

static PetscErrorCode constant(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx) {
    PetscInt d;
    for (d = 0; d < Nc; ++d) {
        u[d] = 1.0;
    }
    return 0;
}

static PetscErrorCode createPressureNullSpace(DM dm, PetscInt ofield, PetscInt nfield, MatNullSpace *nullSpace) {
    Vec vec;

    PetscFunctionBeginUser;
    // determine the number of fields from PETSC
    PetscInt numFields;
    PetscCall(DMGetNumFields(dm, &numFields));

    // Project to the field specified in nfield
    std::vector<PetscErrorCode (*)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *)> funcs(numFields, zero);
    funcs[nfield] = constant;
    PetscCall(DMCreateGlobalVector(dm, &vec));

    // Get the label for this field
    DMLabel label;
    PetscObject field;
    PetscCall(DMGetField(dm, nfield, &label, &field));

    PetscInt ids[1] = {1};
    DMProjectFunctionLabel(dm, 0.0, label, 1, ids, -1, NULL, &funcs[0], NULL, INSERT_VALUES, vec);

    PetscCall(VecNormalize(vec, NULL));
    PetscCall(PetscObjectSetName((PetscObject)vec, "Pressure Null Space"));
    PetscCall(VecViewFromOptions(vec, NULL, "-pressure_nullspace_view"));
    PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_FALSE, PRES, &vec, nullSpace));
    PetscCall(VecDestroy(&vec));
    PetscFunctionReturn(0);
}

/* Make the discrete pressure discretely divergence free */
static PetscErrorCode removeDiscretePressureNullspaceOnTs(TS ts, ablate::finiteElement::LowMachFlowSolver &flow) {
    Vec u;

    DM dm;

    PetscFunctionBegin;
    PetscCall(TSGetDM(ts, &dm));
    PetscCall(TSGetSolution(ts, &u));
    try {
        flow.CompleteFlowInitialization(dm, u);
    } catch (std::exception &exp) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", exp.what());
    }

    PetscFunctionReturn(0);
}

void ablate::finiteElement::LowMachFlowSolver::Initialize() {
    ablate::finiteElement::FiniteElementSolver::Initialize();

    DMSetNullSpaceConstructor(subDomain->GetDM(), PRES, createPressureNullSpace) >> utilities::PetscUtilities::checkError;
    RegisterPreStep([&](TS ts, Solver &) { removeDiscretePressureNullspaceOnTs(ts, *this); });
}

void ablate::finiteElement::LowMachFlowSolver::CompleteFlowInitialization(DM dm, Vec u) {
    MatNullSpace nullsp;

    createPressureNullSpace(dm, PRES, PRES, &nullsp) >> utilities::PetscUtilities::checkError;
    MatNullSpaceRemove(nullsp, u) >> utilities::PetscUtilities::checkError;
    MatNullSpaceDestroy(&nullsp) >> utilities::PetscUtilities::checkError;
}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::finiteElement::LowMachFlowSolver, "incompressible FE flow", ARG(std::string, "id", "the name of the flow field"),
         OPT(ablate::domain::Region, "region", "the region to apply this solver.  Default is entire domain"),
         OPT(ablate::parameters::Parameters, "options", "options for the flow passed directly to PETSc"), ARG(ablate::parameters::Parameters, "parameters", "the flow field parameters"),
         ARG(std::vector<ablate::finiteElement::boundaryConditions::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"),
         ARG(std::vector<ablate::mathFunctions::FieldFunction>, "auxFields", "enables and sets the update functions for the auxFields"));