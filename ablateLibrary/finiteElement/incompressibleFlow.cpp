#include "incompressibleFlow.hpp"
#include <stdexcept>
#include "incompressibleFlow.h"
#include "utilities/petscError.hpp"

ablate::finiteElement::IncompressibleFlow::IncompressibleFlow(std::string solverId, std::string region, std::shared_ptr<parameters::Parameters> options,
                                                              std::shared_ptr<parameters::Parameters> parameters, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization,
                                                              std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                              std::vector<std::shared_ptr<mathFunctions::FieldFunction>> auxiliaryFields,
                                                              std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions)
    : FiniteElement(
          solverId, region, options,
          {
              {.name = "velocity", .prefix = "vel_", .components = {"vel" + domain::FieldDescriptor::DIMENSION}},
              {.name = "pressure", .prefix = "pres_"},
              {.name = "temperature", .prefix = "temp_"},
              {.name = "momentum_source",
               .prefix = "momentum_source_",
               .components = auxiliaryFields.empty() ? std::vector<std::string>{} : std::vector<std::string>{"mom" + domain::FieldDescriptor::DIMENSION},
               .type = domain::FieldType::AUX},
              {.name = "mass_source", .prefix = "mass_source_", .components = auxiliaryFields.empty() ? std::vector<std::string>{} : std::vector<std::string>{"mass"}, .type = domain::FieldType::AUX},
              {.name = "energy_source",
               .prefix = "energy_source_",
               .components = auxiliaryFields.empty() ? std::vector<std::string>{} : std::vector<std::string>{"ener"},
               .type = domain::FieldType::AUX},
          },
          initialization, boundaryConditions, auxiliaryFields, exactSolutions),
      parameters(parameters) {}

void ablate::finiteElement::IncompressibleFlow::Setup() {
    FiniteElement::Setup();

    {
        PetscObject pressure;
        MatNullSpace nullspacePres;

        DMGetField(subDomain->GetDM(), PRES, NULL, &pressure) >> checkError;
        MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nullspacePres) >> checkError;
        PetscObjectCompose(pressure, "nullspace", (PetscObject)nullspacePres) >> checkError;
        MatNullSpaceDestroy(&nullspacePres) >> checkError;
    }

    PetscDS prob;
    DMGetDS(subDomain->GetDM(), &prob) >> checkError;

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
static PetscErrorCode removeDiscretePressureNullspaceOnTs(TS ts, ablate::finiteElement::IncompressibleFlow &flow) {
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
    } catch (std::exception &exp) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exp.what());
    }

    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

void ablate::finiteElement::IncompressibleFlow::Initialize() {
    ablate::finiteElement::FiniteElement::Initialize();

    DMSetNullSpaceConstructor(subDomain->GetDM(), PRES, createPressureNullSpace) >> checkError;
    RegisterPreStep([&](TS ts, Solver &) { removeDiscretePressureNullspaceOnTs(ts, *this); });
}

void ablate::finiteElement::IncompressibleFlow::CompleteFlowInitialization(DM dm, Vec u) {
    MatNullSpace nullsp;

    createPressureNullSpace(dm, PRES, PRES, &nullsp) >> checkError;
    MatNullSpaceRemove(nullsp, u) >> checkError;
    MatNullSpaceDestroy(&nullsp) >> checkError;
}

#include "parser/registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::finiteElement::IncompressibleFlow, "incompressible FE flow", ARG(std::string, "id", "the name of the flow field"),
         OPT(std::string, "region", "the region to apply this solver.  Default is entire domain"), OPT(ablate::parameters::Parameters, "options", "options for the flow passed directly to PETSc"),
         ARG(ablate::parameters::Parameters, "parameters", "the flow field parameters"),
         ARG(std::vector<mathFunctions::FieldFunction>, "initialization", "the solution used to initialize the flow field"),
         ARG(std::vector<finiteElement::boundaryConditions::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"),
         OPT(std::vector<mathFunctions::FieldFunction>, "auxFields", "enables and sets the update functions for the auxFields"),
         OPT(std::vector<mathFunctions::FieldFunction>, "exactSolution", "optional exact solutions that can be used for error calculations"));