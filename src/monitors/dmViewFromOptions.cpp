#include "dmViewFromOptions.hpp"
#include <utilities/petscError.hpp>
#include <utilities/petscOptions.hpp>
#include <utility>
#include "environment/runEnvironment.hpp"
#include "solver/solver.hpp"

ablate::monitors::DmViewFromOptions::DmViewFromOptions(Scope scope, std::string options, const std::string& optionNameIn)
    : petscOptions(nullptr), optionName(optionNameIn.empty() ? "-CallDmViewFromOptions" : optionNameIn), scope(scope) {
    // Set the options
    if (!options.empty()) {
        PetscOptionsCreate(&petscOptions) >> checkError;

        // build the string
        ablate::environment::RunEnvironment::Get().ExpandVariables(options);
        std::string optionString = optionName + " " + options;
        PetscOptionsInsertString(petscOptions, optionString.c_str());
    }
}

ablate::monitors::DmViewFromOptions::DmViewFromOptions(std::string options, std::string optionNameIn) : DmViewFromOptions(Scope::INITIAL, std::move(options), std::move(optionNameIn)) {}

ablate::monitors::DmViewFromOptions::~DmViewFromOptions() {
    if (petscOptions) {
        ablate::utilities::PetscOptionsDestroyAndCheck("DmViewFromOptions", &petscOptions);
    }
}
void ablate::monitors::DmViewFromOptions::Register(std::shared_ptr<solver::Solver> monitorableObject) {
    ablate::monitors::Monitor::Register(monitorableObject);

    if (scope == Scope::INITIAL) {
        // if the scope is initial, dm plex only once during register
        // this probe will only work with fV flow with a single mpi rank for now.  It should be replaced with DMInterpolationEvaluate
        auto flow = std::dynamic_pointer_cast<ablate::solver::Solver>(monitorableObject);
        if (!flow) {
            throw std::invalid_argument("The DmViewFromOptions monitor can only be used with ablate::solver::Solver");
        }

        DMViewFromOptions(flow->GetSubDomain().GetDM()) >> checkError;
    }
}

PetscErrorCode ablate::monitors::DmViewFromOptions::DMViewFromOptions(DM dm) {
    PetscFunctionBeginUser;

    PetscViewer viewer;
    PetscBool flg;
    PetscViewerFormat format;

    PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)dm), petscOptions, nullptr, optionName.c_str(), &viewer, &format, &flg));

    if (flg) {
        PetscCall(PetscViewerPushFormat(viewer, format));
        PetscCall(PetscObjectView((PetscObject)dm, viewer));
        PetscCall(PetscViewerFlush(viewer));
        PetscCall(PetscViewerPopFormat(viewer));
        PetscCall(PetscViewerDestroy(&viewer));
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::monitors::DmViewFromOptions::CallDmViewFromOptions(TS ts, PetscInt, PetscReal, Vec, void* mctx) {
    PetscFunctionBeginUser;
    DM dm;
    PetscCall(TSGetDM(ts, &dm));

    auto monitor = (DmViewFromOptions*)mctx;
    if (monitor->scope == Scope::MONITOR) {
        PetscCall(monitor->DMViewFromOptions(dm));
    }

    PetscFunctionReturn(0);
}
void ablate::monitors::DmViewFromOptions::Modify(DM& dm) { DMViewFromOptions(dm) >> checkError; }

std::ostream& ablate::monitors::operator<<(std::ostream& os, const ablate::monitors::DmViewFromOptions::Scope& v) {
    switch (v) {
        case DmViewFromOptions::Scope::INITIAL:
            return os << "initial";
        case DmViewFromOptions::Scope::MONITOR:
            return os << "monitor";
        default:
            return os;
    }
}

std::istream& ablate::monitors::operator>>(std::istream& is, ablate::monitors::DmViewFromOptions::Scope& v) {
    std::string enumString;
    is >> enumString;

    if (enumString == "initial") {
        v = DmViewFromOptions::Scope::INITIAL;
    } else if (enumString == "monitor") {
        v = DmViewFromOptions::Scope::MONITOR;
    } else {
        throw std::invalid_argument("Unknown norm type " + enumString);
    }
    return is;
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::DmViewFromOptions,
         "replicates the [DMViewFromOptions](https://petsc.org/release/docs/manualpages/Viewer/PetscOptionsGetViewer.html) function in PETSC",
         ENUM(ablate::monitors::DmViewFromOptions::Scope, "scope", "determines if DMViewFromOptions is called initially (initial) or every time step (monitor)"),
         OPT(std::string, "options", "if provided these options are used for the DMView call, otherwise global options is used"),
         OPT(std::string, "optionName", "if provided the optionsName is used for DMViewFromOptions.  Needed if using global options."));

REGISTER(ablate::domain::modifiers::Modifier, ablate::monitors::DmViewFromOptions,
         "replicates the [DMViewFromOptions](https://petsc.org/release/docs/manualpages/Viewer/PetscOptionsGetViewer.html) function in PETSC",
         OPT(std::string, "options", "if provided these options are used for the DMView call, otherwise global options is used"),
         OPT(std::string, "optionName", "if provided the optionsName is used for DMViewFromOptions.  Needed if using global options."));
