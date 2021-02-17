
#include <petsc.h>
#include <memory>
#include "solve/timeStepper.hpp"
#include "mesh/boxMesh.hpp"
#include "parameters/mapParameters.hpp"
#include "flow/lowMachFlow.hpp"
#include "utilities/petscError.hpp"

int main(int argc, char **args) {
    // initialize petsc and mpi
    PetscInitialize(&argc, &args, NULL, NULL) >> ablate::checkError;

    // Create time stepping wrapper
    auto ts = std::make_unique<ablate::solve::TimeStepper>(PETSC_COMM_WORLD, "testTimeStepper", std::map<std::string, std::string>({
                                                                                                    {"ts_dt", "0.1"},
                                                                                                    {"ts_max_steps", "4"}}));

    // Create a mesh
    auto mesh = std::make_shared<ablate::mesh::BoxMesh>(PETSC_COMM_WORLD, "testBoxMesh", std::map<std::string, std::string>({
                                                                                             {"dm_plex_separate_marker", ""},
                                                                                             {"dm_refine", "0"}}), 2);

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
    auto flow = std::make_shared<ablate::flow::LowMachFlow>(mesh, "low mach flow", std::map<std::string, std::string>(), parameters);


    DMComputeExactSolution(mesh->GetDomain(), 0, flow->GetFlowSolution(), NULL) >> ablate::checkError;

    ts->Solve(flow);

    PetscFinalize() >> ablate::checkError;

}