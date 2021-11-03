#include "solver.hpp"
#include <regex>
#include <utilities/petscError.hpp>
#include <utilities/petscOptions.hpp>

ablate::solver::Solver::Solver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options)
    : solverId(solverId), region(region), petscOptions(nullptr) {
    // Set the options
    if (options) {
        PetscOptionsCreate(&petscOptions) >> checkError;
        options->Fill(petscOptions);
    }
}

ablate::solver::Solver::~Solver() {
    if (petscOptions) {
        ablate::utilities::PetscOptionsDestroyAndCheck(GetId(), &petscOptions);
    }
}

void ablate::solver::Solver::Register(std::shared_ptr<ablate::domain::SubDomain> subDomainIn) { subDomain = subDomainIn; }

void ablate::solver::Solver::DecompressFieldFieldDescriptor(std::vector<ablate::domain::FieldDescriptor> &fieldDescriptors) {
    for (auto &field : fieldDescriptors) {
        for (std::size_t c = 0; c < field.components.size(); c++) {
            if (field.components[c].find(domain::FieldDescriptor::DIMENSION) != std::string::npos) {
                auto baseName = field.components[c];

                // Delete this component
                field.components.erase(field.components.begin() + c);

                for (PetscInt d = subDomain->GetDimensions() - 1; d >= 0; d--) {
                    auto newName = std::regex_replace(baseName, std::regex(domain::FieldDescriptor::DIMENSION), std::to_string(d));  // replace 'def' -> 'klm'
                    field.components.insert(field.components.begin() + c, newName);
                }
            }
        }
    }
}
void ablate::solver::Solver::PreStage(TS ts, PetscReal stagetime) {
    for (auto &function : preStageFunctions) {
        function(ts, *this, stagetime);
    }
}
void ablate::solver::Solver::PreStep(TS ts) {
    for (auto &function : preStepFunctions) {
        function(ts, *this);
    }
}
void ablate::solver::Solver::PostStep(TS ts) {
    for (auto &function : postStepFunctions) {
        function(ts, *this);
    }
}
void ablate::solver::Solver::PostEvaluate(TS ts) {
    for (auto &function : postEvaluateFunctions) {
        function(ts, *this);
    }
}

void ablate::solver::Solver::Save(PetscViewer viewer, PetscInt steps, PetscReal time) const {
    auto subDm = subDomain->GetSubDM();
    auto auxDM = subDomain->GetSubAuxDM();
    // If this is the first output, save the mesh
    if (steps == 0) {
        // Print the initial mesh
        DMView(subDm, viewer) >> checkError;
    }

    // set the dm sequence number, because we may be skipping outputs
    DMSetOutputSequenceNumber(subDm, steps, time) >> checkError;
    if (auxDM) {
        DMSetOutputSequenceNumber(auxDM, steps, time) >> checkError;
    }

    // Always save the main flowField
    VecView(subDomain->GetSubSolutionVector(), viewer) >> checkError;

    // If there is aux data output
    if (auto subAuxVector = subDomain->GetSubAuxVector()) {
        // copy over the sequence data from the main dm
        PetscReal dmTime;
        PetscInt dmSequence;
        DMGetOutputSequenceNumber(subDm, &dmSequence, &dmTime) >> checkError;
        DMSetOutputSequenceNumber(auxDM, dmSequence, dmTime) >> checkError;

        Vec auxGlobalField;
        DMGetGlobalVector(auxDM, &auxGlobalField) >> checkError;

        // copy over the name of the auxFieldVector
        const char *tempName;
        PetscObjectGetName((PetscObject)subAuxVector, &tempName) >> checkError;
        PetscObjectSetName((PetscObject)auxGlobalField, tempName) >> checkError;
        DMLocalToGlobal(auxDM, subAuxVector, INSERT_VALUES, auxGlobalField) >> checkError;
        VecView(auxGlobalField, viewer) >> checkError;
        DMRestoreGlobalVector(auxDM, &auxGlobalField) >> checkError;
    }
}

void ablate::solver::Solver::Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    // The only item that needs to be explicitly restored is the flowField
    DMSetOutputSequenceNumber(subDomain->GetDM(), sequenceNumber, time) >> checkError;
    VecLoad(subDomain->GetSolutionVector(), viewer) >> checkError;
}


static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx) {
    PetscInt c;
    for (c = 0; c < Nc; ++c) u[c] = 0.0;
    return 0;
}

PetscErrorCode ablate::solver::Solver::DMPlexInsertBoundaryValues_Plex(DM dm, PetscDS prob, PetscBool insertEssential, Vec locX, PetscReal time, Vec faceGeomFVM, Vec cellGeomFVM,
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

PetscErrorCode ablate::solver::Solver::DMPlexInsertTimeDerivativeBoundaryValues_Plex(DM dm, PetscDS prob, PetscBool insertEssential, Vec locX, PetscReal time, Vec faceGeomFVM,
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
