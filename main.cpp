
#include <petsc.h>
#include <memory>
#include "flow/incompressibleFlow.hpp"
#include "incompressibleFlow.h"
#include "mesh/boxMesh.hpp"
#include "parameters/mapParameters.hpp"
#include "parser/factory.hpp"
#include "parser/registrar.hpp"
#include "solve/timeStepper.hpp"
#include "utilities/petscError.hpp"
#include "utilities/petscOptions.hpp"
#include "parser/listing.h"

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
    ierr = IncompressibleFlow_CompleteFlowInitialization(dm, u);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

int main(int argc, char **args) {
        // initialize petsc and mpi
        PetscInitialize(&argc, &args, NULL, NULL) >> ablate::checkError;

        std::cout << ablate::parser::Listing::Get() << std::endl;


        {
            // -dm_plex_separate_marker
            // -dm_refine 2   -ts_max_steps 30 -ts_dt 0.1 -dm_view hdf5:sol.h5 -num_sol_vec_view_monitor hdf5:sol.h5::append -exact__vec_view hdf5:sol.h5::append
            //
            // -vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1  -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_atol 1.0e-14 -ksp_error_if_not_converged
//            -pc_type
            // fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_0_pc_type lu
            // -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi

            std::map<std::string, std::string> globalParams = {{"dm_plex_separate_marker", ""}};
            ablate::utilities::PetscOptions::Set(globalParams);

            // Create time stepping wrapper
            auto ts = std::make_unique<ablate::solve::TimeStepper>(PETSC_COMM_WORLD,
                                                                   "testTimeStepper",
                                                                   std::map<std::string, std::string>({{"ts_dt", ".1"},
                                                                                                       {"ts_max_steps", "30"},
                                                                                                       {"ksp_type", "fgmres"},
                                                                                                       {"ksp_gmres_restart", "10"},
                                                                                                       {"ksp_rtol", "1.0e-9"},
                                                                                                       {"ksp_atol", "1.0e-14"},
                                                                                                       {"ksp_error_if_not_converged", ""},
                                                                                                       {"pc_type", "fieldsplit"},
                                                                                                       {"pc_fieldsplit_0_fields", "0,2"},
                                                                                                       {"pc_fieldsplit_1_fields", "1"},
                                                                                                       {"pc_fieldsplit_type", "schur"},
                                                                                                       {"pc_fieldsplit_schur_factorization_type", "full"},
                                                                                                       {"fieldsplit_0_pc_type", "lu"},
                                                                                                       {"fieldsplit_pressure_ksp_rtol", "1E-10"},
                                                                                                       {"fieldsplit_pressure_pc_type", "jacobi"}}));

            // Create a mesh
            auto mesh = std::make_shared<ablate::mesh::BoxMesh>(PETSC_COMM_WORLD,
                                                                "testBoxMesh",
                                                                std::map<std::string, std::string>({{"num_sol_vec_view_monitor", "hdf5:sol.h5::append"},
                                                                                                    {"dm_view", "hdf5:sol.h5"},
                                                                                                    {"dm_plex_separate_marker", ""},
                                                                                                    {"dm_refine", "2"},
                                                                                                    {"vel_petscspace_degree", "2"},
                                                                                                    {"pres_petscspace_degree", "1"},
                                                                                                    {"temp_petscspace_degree", "1"}}),
                                                                2);

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

            ts->Solve(flow);
        }
        PetscFinalize() >> ablate::checkError;
}