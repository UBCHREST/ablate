
#include <petsc.h>
#include <memory>
#include "timeStepper.hpp"
#include "boxMesh.hpp"


int main(int argc, char **args) {
    // initialize petsc and mpi
    PetscErrorCode ierr = PetscInitialize(&argc, &args, NULL, NULL);
    CHKERRQ(ierr);

    // Create time stepping wrapper
    auto ts = make_unique<ablate::TimeStepper>(PETSC_COMM_WORLD, "testTimeStepper", std::map<std::string, std::string>());

    // Create a mesh
    auto mesh = make_unique<ablate::mesh::BoxMesh>(PETSC_COMM_WORLD, "testBoxMesh", std::map<std::string, std::string>(), 2);



}