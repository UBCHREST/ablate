#include "fileMesh.hpp"
#include <utilities/petscError.hpp>
#include <utilities/petscOptions.hpp>

ablate::domain::FileMesh::FileMesh(std::string nameIn, std::filesystem::path pathIn, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors,
                                   std::vector<std::shared_ptr<modifiers::Modifier>> modifiers)
    : Domain(ReadDMFromFile(nameIn, pathIn), nameIn, fieldDescriptors, modifiers) {}

ablate::domain::FileMesh::~FileMesh() {
    if (dm) {
        DMDestroy(&dm);
    }
}
DM ablate::domain::FileMesh::ReadDMFromFile(const std::string& name, const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::invalid_argument("Unable to locate file " + path.string());
    }

    DM dm;
    DMPlexCreateFromFile(PETSC_COMM_WORLD, path.c_str(), name.c_str(), PETSC_TRUE, &dm) >> checkError;
    PetscObjectSetName((PetscObject)dm, name.c_str()) >> checkError;
    return dm;
}

#include "registrar.hpp"
REGISTER(ablate::domain::Domain, ablate::domain::FileMesh, "read a DMPlex from a file", ARG(std::string, "name", "the name of the domain/mesh object"),
         ARG(std::filesystem::path, "path", "the path to the mesh file"), OPT(std::vector<ablate::domain::FieldDescriptor>, "fields", "a list of fields/field descriptors"),
         OPT(std::vector<ablate::domain::modifiers::Modifier>, "modifiers", "a list of domain modifier"));