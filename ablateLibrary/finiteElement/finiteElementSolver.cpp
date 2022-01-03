#include "finiteElementSolver.hpp"
#include <petsc.h>
#include <petsc/private/dmpleximpl.h>
#include <petscds.h>
#include <petscfv.h>
#include <utilities/mpiError.hpp>
#include <utilities/petscError.hpp>

ablate::finiteElement::FiniteElementSolver::FiniteElementSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options,
                                                                std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                                std::vector<std::shared_ptr<mathFunctions::FieldFunction>> auxiliaryFields)
    : Solver(solverId, region, options), boundaryConditions(boundaryConditions), auxiliaryFieldsUpdaters(auxiliaryFields) {}

void ablate::finiteElement::FiniteElementSolver::Register(std::shared_ptr<ablate::domain::SubDomain> subDomain) { Solver::Register(subDomain); }

void ablate::finiteElement::FiniteElementSolver::Setup() {
    DM cdm = subDomain->GetDM();

    while (cdm) {
        DMCopyDisc(subDomain->GetDM(), cdm) >> checkError;
        DMGetCoarseDM(cdm, &cdm) >> checkError;
    }

    // Register the aux fields updater if specified
    if (!auxiliaryFieldsUpdaters.empty()) {
        RegisterPreStep([&](TS ts, Solver &) { UpdateAuxFields(ts, *this); });
    }

    // add each boundary condition
    for (auto boundary : boundaryConditions) {
        const auto &fieldId = subDomain->GetField(boundary->GetFieldName());

        // Setup the boundary condition
        boundary->SetupBoundary(subDomain->GetDM(), subDomain->GetDiscreteSystem(), fieldId.id);
    }
}

void ablate::finiteElement::FiniteElementSolver::Initialize() {
    // Initialize the flow field if provided
    this->CompleteFlowInitialization(subDomain->GetDM(), subDomain->GetSolutionVector());
}

void ablate::finiteElement::FiniteElementSolver::UpdateAuxFields(TS ts, ablate::finiteElement::FiniteElementSolver &fe) {
    PetscInt numberAuxFields;
    DMGetNumFields(fe.subDomain->GetAuxDM(), &numberAuxFields) >> checkError;

    // size up the update and context functions
    std::vector<mathFunctions::PetscFunction> auxiliaryFieldFunctions(numberAuxFields, NULL);
    std::vector<void *> auxiliaryFieldContexts(numberAuxFields, NULL);

    // for each given aux field
    for (auto auxFieldDescription : fe.auxiliaryFieldsUpdaters) {
        auto fieldId = fe.subDomain->GetField(auxFieldDescription->GetName());
        auxiliaryFieldContexts[fieldId.id] = auxFieldDescription->GetSolutionField().GetContext();
        auxiliaryFieldFunctions[fieldId.id] = auxFieldDescription->GetSolutionField().GetPetscFunction();
    }

    // get the time at the end of the time step
    PetscReal time = 0;
    PetscReal dt = 0;
    TSGetTime(ts, &time) >> checkError;
    TSGetTimeStep(ts, &dt) >> checkError;

    // Update the source terms
    DMProjectFunctionLocal(fe.subDomain->GetAuxDM(), time + dt, &auxiliaryFieldFunctions[0], &auxiliaryFieldContexts[0], INSERT_ALL_VALUES, fe.subDomain->GetAuxVector()) >> checkError;
}

void ablate::finiteElement::FiniteElementSolver::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) const {
    Solver::Save(viewer, sequenceNumber, time);
}

PetscErrorCode ablate::finiteElement::FiniteElementSolver::ComputeIFunction(PetscReal time, Vec locX, Vec locX_t, Vec locF) {
    PetscFunctionBegin;
    DM plex;
    IS allcellIS;
    PetscErrorCode ierr;

    ierr = DMConvert(subDomain->GetDM(), DMPLEX, &plex);
    CHKERRQ(ierr);
    ierr = DMPlexGetAllCells_Internal(plex, &allcellIS);
    CHKERRQ(ierr);

    IS cellIS;
    PetscFormKey key;
    key.label = subDomain->GetLabel();
    key.value = 0;
    key.field = 0;
    key.part = 0;
    if (!key.label) {
        ierr = PetscObjectReference((PetscObject)allcellIS);
        CHKERRQ(ierr);
        cellIS = allcellIS;
    } else {
        IS pointIS;

        key.value = 1;
        ierr = DMLabelGetStratumIS(key.label, key.value, &pointIS);
        CHKERRQ(ierr);
        ierr = ISIntersect_Caching_Internal(allcellIS, pointIS, &cellIS);
        CHKERRQ(ierr);
        ierr = ISDestroy(&pointIS);
        CHKERRQ(ierr);
    }
    ierr = DMPlexComputeResidual_Internal(plex, key, cellIS, time, locX, locX_t, time, locF, nullptr);
    CHKERRQ(ierr);
    ierr = ISDestroy(&cellIS);
    CHKERRQ(ierr);

    ierr = ISDestroy(&allcellIS);
    CHKERRQ(ierr);
    ierr = DMDestroy(&plex);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteElement::FiniteElementSolver::ComputeIJacobian(PetscReal time, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP) {
    PetscFunctionBeginUser;

    DM plex;
    IS allcellIS;
    PetscBool hasJac, hasPrec;
    PetscErrorCode ierr;

    ierr = DMConvert(subDomain->GetDM(), DMPLEX, &plex);
    ierr = DMPlexGetAllCells_Internal(plex, &allcellIS);
    CHKERRQ(ierr);

    PetscDS ds = subDomain->GetDiscreteSystem();
    IS cellIS;
    PetscFormKey key;
    key.label = subDomain->GetLabel();
    key.value = 0;
    key.field = 0;
    key.part = 0;
    if (!key.label) {
        ierr = PetscObjectReference((PetscObject)allcellIS);
        CHKERRQ(ierr);
        cellIS = allcellIS;
    } else {
        IS pointIS;

        key.value = 1;
        ierr = DMLabelGetStratumIS(key.label, key.value, &pointIS);
        CHKERRQ(ierr);
        ierr = ISIntersect_Caching_Internal(allcellIS, pointIS, &cellIS);
        CHKERRQ(ierr);
        ierr = ISDestroy(&pointIS);
        CHKERRQ(ierr);
    }
    ierr = PetscDSHasJacobian(ds, &hasJac);
    CHKERRQ(ierr);
    ierr = PetscDSHasJacobianPreconditioner(ds, &hasPrec);
    CHKERRQ(ierr);
    if (hasJac && hasPrec) {
        ierr = MatZeroEntries(Jac);
        CHKERRQ(ierr);
    }
    ierr = MatZeroEntries(JacP);
    CHKERRQ(ierr);

    ierr = DMPlexComputeJacobian_Internal(plex, key, cellIS, time, X_tShift, locX, locX_t, Jac, JacP, nullptr);
    CHKERRQ(ierr);
    ierr = ISDestroy(&cellIS);
    CHKERRQ(ierr);

    ierr = ISDestroy(&allcellIS);
    CHKERRQ(ierr);
    ierr = DMDestroy(&plex);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteElement::FiniteElementSolver::ComputeBoundary(PetscReal time, Vec locX, Vec locX_t) {
    PetscFunctionBeginUser;

    DM plex;
    PetscDS ds = subDomain->GetDiscreteSystem();
    PetscErrorCode ierr;
    ierr = DMConvert(subDomain->GetDM(), DMPLEX, &plex);
    CHKERRQ(ierr);

    ierr = ablate::solver::Solver::DMPlexInsertBoundaryValues_Plex(plex, ds, PETSC_TRUE, locX, time, NULL, NULL, NULL);
    CHKERRQ(ierr);
    ierr = ablate::solver::Solver::DMPlexInsertTimeDerivativeBoundaryValues_Plex(plex, ds, PETSC_TRUE, locX_t, time, NULL, NULL, NULL);
    CHKERRQ(ierr);

    ierr = DMDestroy(&plex);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}
