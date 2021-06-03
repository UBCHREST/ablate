#include "incompressibleFlow.hpp"
#include <stdexcept>
#include "incompressibleFlow.h"
#include "parser/registrar.hpp"
#include "utilities/petscError.hpp"

ablate::flow::IncompressibleFlow::IncompressibleFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<parameters::Parameters> options,
                                                     std::vector<std::shared_ptr<FlowFieldSolution>> initialization, std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions,
                                                     std::vector<std::shared_ptr<FlowFieldSolution>> auxiliaryFields)
    : Flow(name, mesh, parameters, options, initialization, boundaryConditions, auxiliaryFields) {
    // Register each field, this order must match the order in LowMachFlowFields enum
    RegisterField({.fieldName = "velocity", .fieldPrefix = "vel_", .components = dim, .fieldType = FieldType::FE});
    RegisterField({.fieldName = "pressure", .fieldPrefix = "pres_", .components = 1, .fieldType = FieldType::FE});
    RegisterField({.fieldName = "temperature", .fieldPrefix = "temp_", .components = 1, .fieldType = FieldType::FE});

    FinalizeRegisterFields();

    DM cdm = dm->GetDomain();
    while (cdm) {
        DMCopyDisc(dm->GetDomain(), cdm) >> checkError;
        DMGetCoarseDM(cdm, &cdm) >> checkError;
    }

    {
        PetscObject pressure;
        MatNullSpace nullspacePres;

        DMGetField(dm->GetDomain(), PRES, NULL, &pressure) >> checkError;
        MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nullspacePres) >> checkError;
        PetscObjectCompose(pressure, "nullspace", (PetscObject)nullspacePres) >> checkError;
        MatNullSpaceDestroy(&nullspacePres) >> checkError;
    }

    // Add in any aux fields the
    if (!auxiliaryFields.empty()) {
        RegisterAuxField({.fieldName = "momentum_source", .fieldPrefix = "momentum_source_", .components = dim, .fieldType = FieldType::FE});
        RegisterAuxField({.fieldName = "mass_source", .fieldPrefix = "mass_source_", .components = 1, .fieldType = FieldType::FE});
        RegisterAuxField({.fieldName = "energy_source", .fieldPrefix = "energy_source_", .components = 1, .fieldType = FieldType::FE});
    }

    PetscDS prob;
    DMGetDS(dm->GetDomain(), &prob) >> checkError;

    // V, W, Q Test Function
    PetscDSSetResidual(prob, VTEST, IncompressibleFlow_vIntegrandTestFunction, IncompressibleFlow_vIntegrandTestGradientFunction) >> checkError;
    PetscDSSetResidual(prob, WTEST, IncompressibleFlow_wIntegrandTestFunction, IncompressibleFlow_wIntegrandTestGradientFunction) >> checkError;
    PetscDSSetResidual(prob, QTEST, IncompressibleFlow_qIntegrandTestFunction, NULL) >> checkError;

    PetscDSSetJacobian(prob, VTEST, VEL, IncompressibleFlow_g0_vu, IncompressibleFlow_g1_vu, NULL, IncompressibleFlow_g3_vu) >> checkError;
    PetscDSSetJacobian(prob, VTEST, PRES, NULL, NULL, IncompressibleFlow_g2_vp, NULL) >> checkError;
    PetscDSSetJacobian(prob, QTEST, VEL, NULL, IncompressibleFlow_g1_qu, NULL, NULL) >> checkError;
    PetscDSSetJacobian(prob, WTEST, VEL, IncompressibleFlow_g0_wu, NULL, NULL, NULL) >> checkError;
    PetscDSSetJacobian(prob, WTEST, TEMP, IncompressibleFlow_g0_wT, IncompressibleFlow_g1_wT, NULL, IncompressibleFlow_g3_wT) >> checkError;

    /* Setup constants */;
    PetscReal parameterArray[TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS];
    parameters->Fill(TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS, incompressibleFlowParametersTypeNames, parameterArray, defaultParameters);
    PetscDSSetConstants(prob, TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS, parameterArray) >> checkError;
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
    PetscErrorCode (*funcs[3])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {zero, zero, zero};
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    if (ofield != PRES) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Nullspace must be for pressure field at correct index, not %D", ofield);
    funcs[nfield] = constant;
    ierr = DMCreateGlobalVector(dm, &vec);
    CHKERRQ(ierr);
    ierr = DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, vec);
    CHKERRQ(ierr);
    ierr = VecNormalize(vec, NULL);
    CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)vec, "Pressure Null Space");
    CHKERRQ(ierr);
    ierr = VecViewFromOptions(vec, NULL, "-pressure_nullspace_view");
    CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_FALSE, PRES, &vec, nullSpace);
    CHKERRQ(ierr);
    ierr = VecDestroy(&vec);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/* Make the discrete pressure discretely divergence free */
static PetscErrorCode removeDiscretePressureNullspaceOnTs(TS ts, ablate::flow::Flow& flow) {
    Vec u;
    PetscErrorCode ierr;
    DM dm;

    PetscFunctionBegin;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = TSGetSolution(ts, &u);
    CHKERRQ(ierr);
    try {
        flow.CompleteFlowInitialization(dm, u);
    }catch(std::exception& exp){
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exp.what());
    }

    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

void ablate::flow::IncompressibleFlow::CompleteProblemSetup(TS ts) {
    ablate::flow::Flow::CompleteProblemSetup(ts);

    DM dm;
    TSGetDM(ts, &dm) >> checkError;
    DMSetNullSpaceConstructor(dm, PRES, createPressureNullSpace) >> checkError;
    preStepFunctions.push_back(removeDiscretePressureNullspaceOnTs);
}

void ablate::flow::IncompressibleFlow::CompleteFlowInitialization(DM dm, Vec u) {
    MatNullSpace nullsp;

    createPressureNullSpace(dm, PRES, PRES, &nullsp)  >> checkError;
    MatNullSpaceRemove(nullsp, u)  >> checkError;
    MatNullSpaceDestroy(&nullsp) >> checkError;

}

//REGISTER(ablate::flow::Flow, ablate::flow::IncompressibleFlow, "incompressible flow", ARG(std::string, "name", "the name of the flow field"), ARG(ablate::mesh::Mesh, "mesh", "the mesh"),
//         ARG(std::map<std::string TMP_COMMA std::string>, "arguments", "arguments to be passed to petsc"), ARG(ablate::parameters::Parameters, "parameters", "incompressible flow parameters"),
//         ARG(std::vector<flow::FlowFieldSolution>, "initialization", "the exact solution used to initialize the flow field"),
//         ARG(std::vector<flow::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"),
//         OPT(std::vector<flow::FlowFieldSolution>, "auxFields", "enables and sets the update functions for the auxFields"));
