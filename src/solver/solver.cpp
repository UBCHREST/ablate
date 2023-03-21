#include "solver.hpp"
#include <petsc/private/dmpleximpl.h>
#include <regex>
#include "utilities/petscUtilities.hpp"

ablate::solver::Solver::Solver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options)
    : solverId(std::move(solverId)), region(std::move(region)), petscOptions(nullptr) {
    // Set the options
    if (options) {
        PetscOptionsCreate(&petscOptions) >> utilities::PetscUtilities::checkError;
        options->Fill(petscOptions);
    }
}

ablate::solver::Solver::~Solver() {
    if (petscOptions) {
        ablate::utilities::PetscUtilities::PetscOptionsDestroyAndCheck(solverId, &petscOptions);
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

    PetscFunctionBegin;
    PetscCall(PetscDSGetNumBoundary(prob, &numBd));
    PetscCall(PetscObjectQuery((PetscObject)locX, "__Vec_bc_zero__", &isZero));
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

        PetscCall(PetscDSGetBoundary(prob, b, &wf, &type, &name, &label, &numids, &ids, &field, &Nc, &comps, &bvfunc, NULL, &ctx));
        if (insertEssential != (type & DM_BC_ESSENTIAL)) continue;
        PetscCall(DMGetField(dm, field, NULL, &obj));
        PetscCall(PetscObjectGetClassId(obj, &id));
        if (id == PETSCFE_CLASSID) {
            switch (type) {
                    /* for FEM, there is no insertion to be done for non-essential boundary conditions */
                case DM_BC_ESSENTIAL: {
                    PetscSimplePointFunc func = (PetscSimplePointFunc)bvfunc;

                    if (isZero) func = zero;
                    PetscCall(DMPlexLabelAddCells(dm, label));
                    PetscCall(DMPlexInsertBoundaryValuesEssential(dm, time, field, Nc, comps, label, numids, ids, func, ctx, locX));
                    PetscCall(DMPlexLabelClearCells(dm, label));
                } break;
                case DM_BC_ESSENTIAL_FIELD: {
                    PetscPointFunc func = (PetscPointFunc)bvfunc;

                    PetscCall(DMPlexLabelAddCells(dm, label));
                    PetscCall(DMPlexInsertBoundaryValuesEssentialField(dm, time, locX, field, Nc, comps, label, numids, ids, func, ctx, locX));
                    PetscCall(DMPlexLabelClearCells(dm, label));
                } break;
                default:
                    break;
            }
        } else if (id == PETSCFV_CLASSID) {
            {
                PetscErrorCode (*func)(PetscReal, const PetscReal *, const PetscReal *, const PetscScalar *, PetscScalar *, void *) =
                    (PetscErrorCode(*)(PetscReal, const PetscReal *, const PetscReal *, const PetscScalar *, PetscScalar *, void *))bvfunc;

                if (!faceGeomFVM) continue;
                PetscCall(DMPlexInsertBoundaryValuesRiemann(dm, time, faceGeomFVM, cellGeomFVM, gradFVM, field, Nc, comps, label, numids, ids, func, ctx, locX));
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

    PetscFunctionBegin;
    if (!locX) PetscFunctionReturn(0);
    PetscCall(PetscDSGetNumBoundary(prob, &numBd));
    PetscCall(PetscObjectQuery((PetscObject)locX, "__Vec_bc_zero__", &isZero));
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

        PetscCall(PetscDSGetBoundary(prob, b, &wf, &type, &name, &label, &numids, &ids, &field, &Nc, &comps, NULL, &bvfunc, &ctx));
        if (insertEssential != (type & DM_BC_ESSENTIAL)) continue;
        PetscCall(DMGetField(dm, field, NULL, &obj));
        PetscCall(PetscObjectGetClassId(obj, &id));
        if (id == PETSCFE_CLASSID) {
            switch (type) {
                    /* for FEM, there is no insertion to be done for non-essential boundary conditions */
                case DM_BC_ESSENTIAL: {
                    PetscSimplePointFunc func_t = (PetscSimplePointFunc)bvfunc;

                    if (isZero) func_t = zero;
                    PetscCall(DMPlexLabelAddCells(dm, label));
                    PetscCall(DMPlexInsertBoundaryValuesEssential(dm, time, field, Nc, comps, label, numids, ids, func_t, ctx, locX));
                    PetscCall(DMPlexLabelClearCells(dm, label));
                } break;
                case DM_BC_ESSENTIAL_FIELD: {
                    PetscPointFunc func_t = (PetscPointFunc)bvfunc;

                    PetscCall(DMPlexLabelAddCells(dm, label));
                    PetscCall(DMPlexInsertBoundaryValuesEssentialField(dm, time, locX, field, Nc, comps, label, numids, ids, func_t, ctx, locX));
                    PetscCall(DMPlexLabelClearCells(dm, label));
                } break;
                default:
                    break;
            }
        } else
            SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, field);
    }
    PetscFunctionReturn(0);
}
