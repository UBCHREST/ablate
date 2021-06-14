#include "dmPlex.hpp"
#include <utilities/petscError.hpp>
#include <utilities/petscOptions.hpp>

ablate::mesh::DMPlex::DMPlex(std::string nameIn, std::shared_ptr<parameters::Parameters> options) : Mesh(nameIn), petscOptions(NULL) {
    // Set the options if provided
    if (options) {
        PetscOptionsCreate(&petscOptions) >> checkError;
        options->Fill(petscOptions);
    }

    DMCreate(PETSC_COMM_WORLD, &dm) >> checkError;
    DMSetType(dm, DMPLEX) >> checkError;
    PetscObjectSetName((PetscObject)dm, name.c_str()) >> checkError;
    PetscObjectSetOptions((PetscObject)dm, petscOptions) >> checkError;
    DMSetFromOptions(dm) >> checkError;
}

ablate::mesh::DMPlex::~DMPlex() {
    if (dm) {
        DMDestroy(&dm);
    }
    if (petscOptions) {
        ablate::utilities::PetscOptionsDestroyAndCheck(name, &petscOptions);
    }
}

#include "parser/registrar.hpp"
REGISTER(ablate::mesh::Mesh, ablate::mesh::DMPlex, "DMPlex that can be set using PETSc options", ARG(std::string, "name", "the mesh dm name"), OPT(ablate::parameters::Parameters, "options", "options used to setup the DMPlex"));