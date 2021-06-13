#include "convergenceCheck.hpp"
#include "parser/registrar.hpp"

void ablate::monitors::flow::ConvergenceCheck::Register(std::shared_ptr<ablate::flow::Flow> flow) { this->flow = flow; }

ablate::monitors::flow::ConvergenceCheck::ConvergenceCheck(int interval) : interval(interval) {}
PetscErrorCode ablate::monitors::flow::ConvergenceCheck::CheckConvergence(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx) {
    PetscFunctionBeginUser;

    PetscErrorCode ierr;

    // Get the monitor
    auto monitor = (ablate::monitors::flow::ConvergenceCheck *)mctx;

    // check to make sure that dmts_check has been set for the options
    PetscOptions options;
    const char *prefix;
    ierr = PetscObjectGetOptions((PetscObject)ts, &options);
    CHKERRQ(ierr);
    ierr = TSGetOptionsPrefix(ts, &prefix);
    CHKERRQ(ierr);

    PetscBool dmtsCheckEnabled;
    PetscOptionsHasName(options, prefix, "-dmts_check", &dmtsCheckEnabled);
    CHKERRQ(ierr);
    if (!dmtsCheckEnabled) {
        PetscOptionsPrefixPush(options, prefix);
        CHKERRQ(ierr);
        PetscOptionsSetValue(options, "-dmts_check", NULL);
        CHKERRQ(ierr);
        PetscOptionsPrefixPop(options);
        CHKERRQ(ierr);
    }

    // check to see if check convergence;
    bool check = steps == 0;
    if (!check && monitor->interval > 1) {
        check = steps % monitor->interval == 0;
    }

    if (check) {
        MPI_Comm comm;
        PetscObjectGetComm((PetscObject)ts, &comm);
        CHKERRQ(ierr);
        ierr = PetscPrintf(comm, "////////////////////////\n");
        CHKERRQ(ierr);
        ierr = PetscPrintf(comm, "%s Convergence at t:%f n:%d\n", monitor->flow->GetName().c_str(), time, steps);
        CHKERRQ(ierr);
        ierr = DMTSCheckFromOptions(ts, u);
        CHKERRQ(ierr);
        ierr = PetscPrintf(comm, "////////////////////////\n");
        CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}

REGISTER(ablate::monitors::flow::FlowMonitor, ablate::monitors::flow::ConvergenceCheck, "checks the convergence of the dm/ts system",
         ARG(int, "interval", "the time stepping interval for convergence checking where 0 results in checking the first timestep only"));
