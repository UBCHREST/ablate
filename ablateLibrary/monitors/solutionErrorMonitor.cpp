#include "solutionErrorMonitor.hpp"
#include <monitors/logs/stdOut.hpp>
#include <utilities/petscError.hpp>
#include "mathFunctions/mathFunction.hpp"

ablate::monitors::SolutionErrorMonitor::SolutionErrorMonitor(ablate::monitors::SolutionErrorMonitor::Scope errorScope, ablate::monitors::SolutionErrorMonitor::Norm normType,
                                                             std::shared_ptr<logs::Log> logIn)
    : errorScope(errorScope), normType(normType), log(logIn ? logIn : std::make_shared<logs::StdOut>()) {}

PetscErrorCode ablate::monitors::SolutionErrorMonitor::MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    DM dm;
    PetscDS ds;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = DMGetDS(dm, &ds);
    CHKERRQ(ierr);

    // Check for the number of DS, this should be relaxed
    PetscInt numberDS;
    ierr = DMGetNumDS(dm, &numberDS);
    CHKERRQ(ierr);
    if (numberDS > 1) {
        SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "This monitor only supports a single DS in a DM");
    }

    // Get the number of fields
    PetscInt numberOfFields;
    ierr = PetscDSGetNumFields(ds, &numberOfFields);
    CHKERRQ(ierr);
    PetscInt* numberComponentsPerField;
    ierr = PetscDSGetComponents(ds, &numberComponentsPerField);
    CHKERRQ(ierr);

    SolutionErrorMonitor* errorMonitor = (SolutionErrorMonitor*)ctx;

    // if this is the first time step init the log
    if (!errorMonitor->log->Initialized()) {
        errorMonitor->log->Initialize(PetscObjectComm((PetscObject)dm));
    }

    std::vector<PetscReal> ferrors;
    try {
        ferrors = errorMonitor->ComputeError(ts, crtime, u);
    } catch (std::exception& exception) {
        SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_LIB, exception.what());
    }

    // get the error type
    std::stringstream errorTypeStream;
    errorTypeStream << errorMonitor->normType;
    auto errorTypeName = errorTypeStream.str();
    // Change the output depending upon type
    switch (errorMonitor->errorScope) {
        case Scope::VECTOR:
            errorMonitor->log->Printf("Timestep: %04d time = %-8.4g \t %s", (int)step, (double)crtime, errorTypeName.c_str());
            errorMonitor->log->Print("error", ferrors, "%2.3g");
            errorMonitor->log->Print("\n");
            break;
        case Scope::COMPONENT: {
            errorMonitor->log->Printf("Timestep: %04d time = %-8.4g \t %s error:\n", (int)step, (double)crtime, errorTypeName.c_str());
            PetscInt fieldOffset = 0;
            for (PetscInt f = 0; f < numberOfFields; f++) {
                PetscObject field;
                ierr = DMGetField(dm, f, NULL, &field);
                CHKERRQ(ierr);
                const char* name;
                PetscObjectGetName((PetscObject)field, &name);

                errorMonitor->log->Print("\t ");
                errorMonitor->log->Print(name, numberComponentsPerField[f], &ferrors[fieldOffset], "%2.3g");
                errorMonitor->log->Print("\n");
                fieldOffset += numberComponentsPerField[f];
            }
        } break;
        default: {
            SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_LIB, "Unknown error scope");
        }
    }

    PetscFunctionReturn(0);
}

std::vector<PetscReal> ablate::monitors::SolutionErrorMonitor::ComputeError(TS ts, PetscReal time, Vec u) {
    DM dm;
    PetscDS ds;
    TSGetDM(ts, &dm) >> checkError;
    DMGetDS(dm, &ds) >> checkError;

    // Get the number of fields
    PetscInt numberOfFields;
    PetscDSGetNumFields(ds, &numberOfFields) >> checkError;
    PetscInt* numberComponentsPerField;
    PetscDSGetComponents(ds, &numberComponentsPerField) >> checkError;

    // compute the total number of components
    PetscInt totalComponents = 0;

    // Get the exact funcs and context
    std::vector<ablate::mathFunctions::PetscFunction> exactFuncs(numberOfFields);
    std::vector<void*> exactCtxs(numberOfFields);
    for (auto f = 0; f < numberOfFields; ++f) {
        PetscDSGetExactSolution(ds, f, &exactFuncs[f], &exactCtxs[f]) >> checkError;
        if (!exactFuncs[f]) {
            throw std::invalid_argument("The exact solution has not set");
        }
        totalComponents += numberComponentsPerField[f];
    }

    // Create an vector to hold the exact solution
    Vec exactVec;
    VecDuplicate(u, &exactVec) >> checkError;
    DMProjectFunction(dm, time, &exactFuncs[0], &exactCtxs[0], INSERT_ALL_VALUES, exactVec) >> checkError;

    // Compute the error
    VecAXPY(exactVec, -1.0, u) >> checkError;

    // If we treat this as a single vector or multiple components change how this is done
    totalComponents = errorScope == Scope::VECTOR ? 1 : totalComponents;

    // Update the block size
    VecSetBlockSize(exactVec, totalComponents) >> checkError;

    // Compute the l2 errors
    std::vector<PetscReal> ferrors(totalComponents);
    NormType petscNormType;
    switch (normType) {
        case Norm::L1_NORM:
        case Norm::L1:
            petscNormType = NORM_1;
            break;
        case Norm::L2_NORM:
        case Norm::L2:
            petscNormType = NORM_2;
            break;
        case Norm::LINF:
            petscNormType = NORM_INFINITY;
            break;
        default:
            std::stringstream error;
            error << "Unable to process norm type " << normType;
            throw std::invalid_argument(error.str());
    }

    // compute the norm along the stride
    VecStrideNormAll(exactVec, petscNormType, &ferrors[0]) >> checkError;

    // normalize the error if _norm
    if (normType == Norm::L1_NORM) {
        PetscInt size;
        VecGetSize(exactVec, &size);
        PetscReal factor = (1.0 / (size / totalComponents));
        for (PetscInt c = 0; c < totalComponents; c++) {
            ferrors[c] *= factor;
        }
    }
    if (normType == Norm::L2_NORM) {
        PetscInt size;
        VecGetSize(exactVec, &size);
        PetscReal factor = PetscSqrtReal(1.0 / (size / totalComponents));
        for (PetscInt c = 0; c < totalComponents; c++) {
            ferrors[c] *= factor;
        }
    }

    VecDestroy(&exactVec) >> checkError;
    return ferrors;
}

std::ostream& ablate::monitors::operator<<(std::ostream& os, const ablate::monitors::SolutionErrorMonitor::Scope& v) {
    switch (v) {
        case SolutionErrorMonitor::Scope::VECTOR:
            return os << "vector";
        case SolutionErrorMonitor::Scope::COMPONENT:
            return os << "component";
        default:
            return os;
    }
}

std::istream& ablate::monitors::operator>>(std::istream& is, ablate::monitors::SolutionErrorMonitor::Scope& v) {
    std::string enumString;
    is >> enumString;

    if (enumString == "vector") {
        v = SolutionErrorMonitor::Scope::VECTOR;
    } else if (enumString == "component") {
        v = SolutionErrorMonitor::Scope::COMPONENT;
    } else {
        throw std::invalid_argument("Unknown Scope type " + enumString);
    }
    return is;
}

std::ostream& ablate::monitors::operator<<(std::ostream& os, const ablate::monitors::SolutionErrorMonitor::Norm& v) {
    switch (v) {
        case SolutionErrorMonitor::Norm::L1:
            return os << "l1";
        case SolutionErrorMonitor::Norm::L1_NORM:
            return os << "l1_norm";
        case SolutionErrorMonitor::Norm::L2:
            return os << "l2";
        case SolutionErrorMonitor::Norm::LINF:
            return os << "linf";
        case SolutionErrorMonitor::Norm::L2_NORM:
            return os << "l2_norm";
        default:
            return os;
    }
}

std::istream& ablate::monitors::operator>>(std::istream& is, ablate::monitors::SolutionErrorMonitor::Norm& v) {
    std::string enumString;
    is >> enumString;

    if (enumString == "l2") {
        v = SolutionErrorMonitor::Norm::L2;
    } else if (enumString == "linf") {
        v = SolutionErrorMonitor::Norm::LINF;
    } else if (enumString == "l2_norm") {
        v = SolutionErrorMonitor::Norm::L2_NORM;
    } else if (enumString == "l1_norm") {
        v = SolutionErrorMonitor::Norm::L1_NORM;
    } else if (enumString == "l1") {
        v = SolutionErrorMonitor::Norm::L1;
    } else {
        throw std::invalid_argument("Unknown norm type " + enumString);
    }
    return is;
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::SolutionErrorMonitor, "Computes and reports the error every time step",
         ENUM(ablate::monitors::SolutionErrorMonitor::Scope, "scope", "how the error should be calculated ('vector', 'component')"),
         ENUM(ablate::monitors::SolutionErrorMonitor::Norm, "type", "norm type ('l1','l1_norm','l2', 'linf', 'l2_norm')"), OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));
