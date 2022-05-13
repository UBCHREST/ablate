#include "solver.hpp"
#include <petsc/private/dmpleximpl.h>
#include <regex>
#include <utilities/petscError.hpp>
#include <utilities/petscOptions.hpp>

ablate::solver::Solver::Solver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options)
    : solverId(std::move(solverId)), region(std::move(region)), petscOptions(nullptr) {
    // Set the options
    if (options) {
        PetscOptionsCreate(&petscOptions) >> checkError;
        options->Fill(petscOptions);
    }
}

ablate::solver::Solver::~Solver() {
    if (petscOptions) {
        ablate::utilities::PetscOptionsDestroyAndCheck(solverId, &petscOptions);
    }
}

void ablate::solver::Solver::Register(std::shared_ptr<ablate::domain::SubDomain> subDomainIn) { subDomain = std::move(subDomainIn); }

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

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx) {
    PetscInt c;
    for (c = 0; c < Nc; ++c) u[c] = 0.0;
    return 0;
}

PetscErrorCode ablate::solver::Solver::DMPlexInsertBoundaryValues_Plex(DM dm, PetscDS prob, PetscBool insertEssential, Vec locX, PetscReal time, Vec faceGeomFVM, Vec cellGeomFVM, Vec gradFVM) {
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
            SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, field);
    }
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::solver::Solver::DMPlexInsertTimeDerivativeBoundaryValues_Plex(DM dm, PetscDS prob, PetscBool insertEssential, Vec locX, PetscReal time, Vec faceGeomFVM, Vec cellGeomFVM,
                                                                                     Vec gradFVM) {
    PetscObject isZero;
    PetscInt numBd, b;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (!locX) PetscFunctionReturn(0);
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
            SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, field);
    }
    PetscFunctionReturn(0);
}

void ablate::solver::Solver::GetCellRange(Range &cellRange) const {
    // Start out getting all the cells
    PetscInt depth;
    DMPlexGetDepth(subDomain->GetDM(), &depth) >> checkError;
    GetRange(depth, cellRange);
}

void ablate::solver::Solver::GetFaceRange(Range &faceRange) const {
    // Start out getting all the faces
    PetscInt depth;
    DMPlexGetDepth(subDomain->GetDM(), &depth) >> checkError;
    GetRange(depth - 1, faceRange);
}

void ablate::solver::Solver::GetRange(PetscInt depth, Range &faceRange) const {
    // Start out getting all of the points
    IS allPointIS;
    DMGetStratumIS(subDomain->GetDM(), "dim", depth, &allPointIS) >> checkError;
    if (!allPointIS) {
        DMGetStratumIS(subDomain->GetDM(), "depth", depth, &allPointIS) >> checkError;
    }

    // If there is a label for this solver, get only the parts of the mesh that here
    if (region) {
        DMLabel label;
        DMGetLabel(subDomain->GetDM(), region->GetName().c_str(), &label);

        IS labelIS;
        DMLabelGetStratumIS(label, GetRegion()->GetValue(), &labelIS) >> checkError;
        ISIntersect_Caching_Internal(allPointIS, labelIS, &faceRange.is) >> checkError;
        ISDestroy(&labelIS) >> checkError;
    } else {
        PetscObjectReference((PetscObject)allPointIS) >> checkError;
        faceRange.is = allPointIS;
    }

    // Get the point range
    if (faceRange.is == nullptr) {
        // There are no points in this region, so skip
        faceRange.start = 0;
        faceRange.end = 0;
        faceRange.points = nullptr;
    } else {
        // Get the range
        ISGetPointRange(faceRange.is, &faceRange.start, &faceRange.end, &faceRange.points) >> checkError;
    }

    // Clean up the allCellIS
    ISDestroy(&allPointIS) >> checkError;
}

void ablate::solver::Solver::RestoreRange(Range &range) const {
    if (range.is) {
        ISRestorePointRange(range.is, &range.start, &range.end, &range.points) >> checkError;
        ISDestroy(&range.is) >> checkError;
    }
}
ablate::solver::Solver::Solver() {}
