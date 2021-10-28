#include "fileMesh.hpp"
#include <utilities/petscError.hpp>
#include <utilities/petscOptions.hpp>
ablate::domain::FileMesh::FileMesh(std::string nameIn, std::filesystem::path pathIn, std::shared_ptr<parameters::Parameters> options, std::vector<std::shared_ptr<modifier::Modifier>> modifiers)
    : Domain(nameIn, modifiers), path(pathIn), petscOptions(NULL) {
    // Set the options if provided
    if (options) {
        PetscOptionsCreate(&petscOptions) >> checkError;
        options->Fill(petscOptions);
    }

    DMPlexCreateFromFile(PETSC_COMM_WORLD, path.c_str(), PETSC_TRUE, &dm) >> checkError;
    PetscObjectSetName((PetscObject)dm, name.c_str()) >> checkError;
    PetscObjectSetOptions((PetscObject)dm, petscOptions) >> checkError;
    DMSetFromOptions(dm) >> checkError;
}

ablate::domain::FileMesh::~FileMesh() {
    if (dm) {
        DMDestroy(&dm);
    }
    if (petscOptions) {
        ablate::utilities::PetscOptionsDestroyAndCheck(name, &petscOptions);
    }
}

#include "parser/registrar.hpp"
REGISTER(ablate::domain::Domain, ablate::domain::FileMesh, "read a DMPlex from a file", ARG(std::string, "name", "the name of the domain/mesh object"),
         ARG(std::filesystem::path, "path", "the path to the mesh file"), OPT(ablate::parameters::Parameters, "options", "any PETSc options"),
         OPT(std::vector<domain::modifier::Modifier>, "modifiers", "a list of domain modifier"));