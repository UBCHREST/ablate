
#include <petsc.h>
#include <memory>
#include "solve/timeStepper.hpp"
#include "mesh/boxMesh.hpp"
#include "parameters/mapParameters.hpp"
#include "flow/incompressibleFlow.hpp"
#include "utilities/petscError.hpp"
#include "incompressibleFlow.h"

PetscErrorCode SetInitialConditions(TS ts, Vec u) {
    DM dm;
    PetscReal t;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = TSGetTime(ts, &t);
    CHKERRQ(ierr);

    // This function Tags the u vector as the exact solution.  We need to copy the values to prevent this.
    Vec e;
    ierr = VecDuplicate(u, &e);
    CHKERRQ(ierr);
    ierr = DMComputeExactSolution(dm, t, e, NULL);
    CHKERRQ(ierr);
    ierr = VecCopy(e, u);
    CHKERRQ(ierr);
    ierr = VecDestroy(&e);
    CHKERRQ(ierr);

    // get the flow to apply the completeFlowInitialization method
    ierr = IncompressibleFlow_CompleteFlowInitialization(dm, u);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

int main(int argc, char **args) {
    // initialize petsc and mpi
    PetscInitialize(&argc, &args, NULL, NULL) >> ablate::checkError;
    {
        // Create time stepping wrapper
        auto ts = std::make_unique<ablate::solve::TimeStepper>(PETSC_COMM_WORLD, "testTimeStepper", std::map<std::string, std::string>({{"ts_dt", ".1"}, {"ts_max_steps", "1000"}}));

        // Create a mesh
        auto mesh = std::make_shared<ablate::mesh::BoxMesh>(PETSC_COMM_WORLD, "testBoxMesh", std::map<std::string, std::string>(/*{{"dm_view", "hdf5:sol.h5"},{"dm_plex_separate_marker", ""}, {"dm_refine", "1"}}*/), 2);

        std::map<std::string, std::string> values = {{"strouhal", "1.0"},
                                                     {"reynolds", "1.0"},
                                                     {"froude", "1.0"},
                                                     {"peclet", "1.0"},
                                                     {"heatRelease", "1.0"},
                                                     {"gamma", "1.0"},
                                                     {"pth", "1.0"},
                                                     {"mu", "1.0"},
                                                     {"k", "1.0"},
                                                     {"cp", "1.0"},
                                                     {"cp", "1.0"},
                                                     {"beta", "1.0"},
                                                     {"gravityDirection", "1"}};
        auto parameters = std::make_shared<ablate::parameters::MapParameters>(values);

        // Create a low flow
        auto flow = std::make_shared<ablate::flow::IncompressibleFlow>(mesh, "low mach flow", std::map<std::string, std::string>(), parameters);

        // Set initial conditions from the exact solution
        TSSetComputeInitialCondition(ts->GetTS(), SetInitialConditions);

        ts->Solve(flow);
    }
    PetscFinalize() >> ablate::checkError;

}