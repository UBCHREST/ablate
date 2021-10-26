#include "finiteElement.hpp"
#include <petsc.h>
#include <petsc/private/dmpleximpl.h>
#include <petscds.h>
#include <petscfv.h>
#include <utilities/mpiError.hpp>
#include <utilities/petscError.hpp>

ablate::finiteElement::FiniteElement::FiniteElement(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options,
                                                    std::vector<ablate::domain::FieldDescriptor> fieldDescriptors, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization,
                                                    std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                    std::vector<std::shared_ptr<mathFunctions::FieldFunction>> auxiliaryFields,
                                                    std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolution)
    : Solver(solverId, region, options),
      fieldDescriptors(fieldDescriptors),
      initialization(initialization),
      boundaryConditions(boundaryConditions),
      auxiliaryFieldsUpdaters(auxiliaryFields),
      exactSolutions(exactSolution) {}

void ablate::finiteElement::FiniteElement::Register(std::shared_ptr<ablate::domain::SubDomain> subDomain) {
    Solver::Register(subDomain);
    Solver::DecompressFieldFieldDescriptor(fieldDescriptors);

    // initialize each field
    for (const auto &field : fieldDescriptors) {
        if (!field.components.empty()) {
            RegisterFiniteElementField(field);
        }
    }
}

void ablate::finiteElement::FiniteElement::Setup() {
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
        const auto &fieldId = subDomain->GetSolutionField(boundary->GetFieldName());

        // Setup the boundary condition
        boundary->SetupBoundary(subDomain->GetDM(), subDomain->GetDiscreteSystem(), fieldId.id);
    }
}

void ablate::finiteElement::FiniteElement::Initialize() {
    // Initialize the flow field if provided
    subDomain->ProjectFieldFunctions(initialization, subDomain->GetSolutionVector());
    this->CompleteFlowInitialization(subDomain->GetDM(), subDomain->GetSolutionVector());

    // if an exact solution has been provided register it
    auto prob = subDomain->GetDiscreteSystem();
    for (const auto &exactSolution : exactSolutions) {
        auto fieldId = subDomain->GetField(exactSolution->GetName());

        // Get the current field type
        if (exactSolution->HasSolutionField()) {
            PetscDSSetExactSolution(prob, fieldId.id, exactSolution->GetSolutionField().GetPetscFunction(), exactSolution->GetSolutionField().GetContext()) >> checkError;
        }
        if (exactSolution->HasTimeDerivative()) {
            PetscDSSetExactSolutionTimeDerivative(prob, fieldId.id, exactSolution->GetTimeDerivative().GetPetscFunction(), exactSolution->GetTimeDerivative().GetContext()) >> checkError;
        }
    }
}

void ablate::finiteElement::FiniteElement::UpdateAuxFields(TS ts, ablate::finiteElement::FiniteElement &fe) {
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

void ablate::finiteElement::FiniteElement::RegisterFiniteElementField(const ablate::domain::FieldDescriptor &fieldDescriptor) {
    // determine if it a simplex element and the number of dimensions
    DMPolytopeType ct;
    PetscInt cStart;
    DMPlexGetHeightStratum(subDomain->GetDM(), 0, &cStart, NULL) >> checkError;
    DMPlexGetCellType(subDomain->GetDM(), cStart, &ct) >> checkError;
    PetscInt simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1 ? PETSC_TRUE : PETSC_FALSE;
    PetscInt simplexGlobal;

    // Assume true if any rank says true
    MPI_Allreduce(&simplex, &simplexGlobal, 1, MPIU_INT, MPI_MAX, subDomain->GetComm()) >> checkMpiError;
    // create a petsc fe
    PetscFE petscFE;
    PetscFECreateDefault(PetscObjectComm((PetscObject)subDomain->GetDM()),
                         subDomain->GetDimensions(),
                         fieldDescriptor.components.size(),
                         simplexGlobal ? PETSC_TRUE : PETSC_FALSE,
                         fieldDescriptor.prefix.c_str(),
                         PETSC_DEFAULT,
                         &petscFE) >>
        checkError;
    PetscObjectSetName((PetscObject)petscFE, fieldDescriptor.name.c_str()) >> checkError;
    PetscObjectSetOptions((PetscObject)petscFE, petscOptions) >> checkError;

    // If this is not the first field, copy the quadrature locations
    if (subDomain->GetNumberFields() > 0) {
        PetscFE referencePetscFE = (PetscFE)subDomain->GetPetscFieldObject(subDomain->GetField(0));
        PetscFECopyQuadrature(referencePetscFE, petscFE) >> checkError;
    }

    // Register the field with the subDomain
    subDomain->RegisterField(fieldDescriptor, (PetscObject)petscFE);
    PetscFEDestroy(&petscFE) >> checkError;
}

void ablate::finiteElement::FiniteElement::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) const {
    Solver::Save(viewer, sequenceNumber, time);

    if (!exactSolutions.empty()) {
        Vec exactVec;
        DMGetGlobalVector(subDomain->GetSubDM(), &exactVec) >> checkError;

        subDomain->ProjectFieldFunctions(exactSolutions, exactVec, time);

        PetscObjectSetName((PetscObject)exactVec, "exact") >> checkError;
        VecView(exactVec, viewer) >> checkError;
        DMRestoreGlobalVector(subDomain->GetSubDM(), &exactVec) >> checkError;
    }
}

PetscErrorCode ablate::finiteElement::FiniteElement::ComputeIFunction(PetscReal time, Vec locX, Vec locX_t, Vec locF) {
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

PetscErrorCode ablate::finiteElement::FiniteElement::ComputeIJacobian(PetscReal time, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP) {
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

PetscErrorCode ablate::finiteElement::FiniteElement::ComputeBoundary(PetscReal time, Vec locX, Vec locX_t) {
    PetscFunctionBeginUser;

    DM plex;
    PetscDS ds = subDomain->GetDiscreteSystem();
    PetscErrorCode ierr;
    ierr = DMConvert(subDomain->GetDM(), DMPLEX, &plex);
    CHKERRQ(ierr);

    ierr = ablate::finiteElement::FiniteElement::DMPlexInsertBoundaryValues_Plex(plex, ds, PETSC_TRUE, locX, time, NULL, NULL, NULL);
    CHKERRQ(ierr);
    ierr = ablate::finiteElement::FiniteElement::DMPlexInsertTimeDerivativeBoundaryValues_Plex(plex, ds, PETSC_TRUE, locX_t, time, NULL, NULL, NULL);
    CHKERRQ(ierr);

    ierr = DMDestroy(&plex);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx) {
    PetscInt c;
    for (c = 0; c < Nc; ++c) u[c] = 0.0;
    return 0;
}

PetscErrorCode ablate::finiteElement::FiniteElement::DMPlexInsertBoundaryValues_Plex(DM dm, PetscDS prob, PetscBool insertEssential, Vec locX, PetscReal time, Vec faceGeomFVM, Vec cellGeomFVM,
                                                                                     Vec gradFVM) {
    PetscObject isZero;
    PetscInt numBd, b;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscDSGetNumBoundary(prob, &numBd);
    CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)locX, "__Vec_bc_zero__", &isZero);
    CHKERRQ(ierr);
    for (b = 0; b < numBd; ++b) {
        PetscWeakForm wf;
        DMBoundaryConditionType type;
        const char *name;
        DMLabel label;
        PetscInt field, Nc;
        const PetscInt *comps;
        PetscObject obj;
        PetscClassId id;
        void (*bvfunc)(void);
        PetscInt numids;
        const PetscInt *ids;
        void *ctx;

        ierr = PetscDSGetBoundary(prob, b, &wf, &type, &name, &label, &numids, &ids, &field, &Nc, &comps, &bvfunc, NULL, &ctx);
        CHKERRQ(ierr);
        if (insertEssential != (type & DM_BC_ESSENTIAL)) continue;
        ierr = DMGetField(dm, field, NULL, &obj);
        CHKERRQ(ierr);
        ierr = PetscObjectGetClassId(obj, &id);
        CHKERRQ(ierr);
        if (id == PETSCFE_CLASSID) {
            switch (type) {
                    /* for FEM, there is no insertion to be done for non-essential boundary conditions */
                case DM_BC_ESSENTIAL: {
                    PetscSimplePointFunc func = (PetscSimplePointFunc)bvfunc;

                    if (isZero) func = zero;
                    ierr = DMPlexLabelAddCells(dm, label);
                    CHKERRQ(ierr);
                    ierr = DMPlexInsertBoundaryValuesEssential(dm, time, field, Nc, comps, label, numids, ids, func, ctx, locX);
                    CHKERRQ(ierr);
                    ierr = DMPlexLabelClearCells(dm, label);
                    CHKERRQ(ierr);
                } break;
                case DM_BC_ESSENTIAL_FIELD: {
                    PetscPointFunc func = (PetscPointFunc)bvfunc;

                    ierr = DMPlexLabelAddCells(dm, label);
                    CHKERRQ(ierr);
                    ierr = DMPlexInsertBoundaryValuesEssentialField(dm, time, locX, field, Nc, comps, label, numids, ids, func, ctx, locX);
                    CHKERRQ(ierr);
                    ierr = DMPlexLabelClearCells(dm, label);
                    CHKERRQ(ierr);
                } break;
                default:
                    break;
            }
        } else if (id == PETSCFV_CLASSID) {
            {
                PetscErrorCode (*func)(PetscReal, const PetscReal *, const PetscReal *, const PetscScalar *, PetscScalar *, void *) =
                    (PetscErrorCode(*)(PetscReal, const PetscReal *, const PetscReal *, const PetscScalar *, PetscScalar *, void *))bvfunc;

                if (!faceGeomFVM) continue;
                ierr = DMPlexInsertBoundaryValuesRiemann(dm, time, faceGeomFVM, cellGeomFVM, gradFVM, field, Nc, comps, label, numids, ids, func, ctx, locX);
                CHKERRQ(ierr);
            }
        } else
            SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
    }
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteElement::FiniteElement::DMPlexInsertTimeDerivativeBoundaryValues_Plex(DM dm, PetscDS prob, PetscBool insertEssential, Vec locX, PetscReal time, Vec faceGeomFVM,
                                                                                                   Vec cellGeomFVM, Vec gradFVM) {
    PetscObject isZero;
    PetscInt numBd, b;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (!locX) PetscFunctionReturn(0);
    ierr = DMGetDS(dm, &prob);
    CHKERRQ(ierr);
    ierr = PetscDSGetNumBoundary(prob, &numBd);
    CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)locX, "__Vec_bc_zero__", &isZero);
    CHKERRQ(ierr);
    for (b = 0; b < numBd; ++b) {
        PetscWeakForm wf;
        DMBoundaryConditionType type;
        const char *name;
        DMLabel label;
        PetscInt field, Nc;
        const PetscInt *comps;
        PetscObject obj;
        PetscClassId id;
        PetscInt numids;
        const PetscInt *ids;
        void (*bvfunc)(void);
        void *ctx;

        ierr = PetscDSGetBoundary(prob, b, &wf, &type, &name, &label, &numids, &ids, &field, &Nc, &comps, NULL, &bvfunc, &ctx);
        CHKERRQ(ierr);
        if (insertEssential != (type & DM_BC_ESSENTIAL)) continue;
        ierr = DMGetField(dm, field, NULL, &obj);
        CHKERRQ(ierr);
        ierr = PetscObjectGetClassId(obj, &id);
        CHKERRQ(ierr);
        if (id == PETSCFE_CLASSID) {
            switch (type) {
                    /* for FEM, there is no insertion to be done for non-essential boundary conditions */
                case DM_BC_ESSENTIAL: {
                    PetscSimplePointFunc func_t = (PetscSimplePointFunc)bvfunc;

                    if (isZero) func_t = zero;
                    ierr = DMPlexLabelAddCells(dm, label);
                    CHKERRQ(ierr);
                    ierr = DMPlexInsertBoundaryValuesEssential(dm, time, field, Nc, comps, label, numids, ids, func_t, ctx, locX);
                    CHKERRQ(ierr);
                    ierr = DMPlexLabelClearCells(dm, label);
                    CHKERRQ(ierr);
                } break;
                case DM_BC_ESSENTIAL_FIELD: {
                    PetscPointFunc func_t = (PetscPointFunc)bvfunc;

                    ierr = DMPlexLabelAddCells(dm, label);
                    CHKERRQ(ierr);
                    ierr = DMPlexInsertBoundaryValuesEssentialField(dm, time, locX, field, Nc, comps, label, numids, ids, func_t, ctx, locX);
                    CHKERRQ(ierr);
                    ierr = DMPlexLabelClearCells(dm, label);
                    CHKERRQ(ierr);
                } break;
                default:
                    break;
            }
        } else
            SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", field);
    }
    PetscFunctionReturn(0);
}