#include "boxMesh.hpp"
#include "../utilities/petscError.hpp"
#include "parser/registrar.hpp"

ablate::mesh::BoxMesh::BoxMesh(std::string name, std::map<std::string, std::string> arguments, int dimensions) : Mesh(PETSC_COMM_WORLD, name, arguments) {
    DMPlexCreateBoxMesh(comm, dimensions, PETSC_TRUE, NULL, NULL, NULL, NULL, PETSC_TRUE, &dm) >> checkError;
    DMSetOptionsPrefix(dm, name.c_str()) >> checkError;
    DMSetFromOptions(dm) >> checkError;
    DMViewFromOptions(dm, NULL, "-dm_view") >> checkError;
}

REGISTER(ablate::mesh::Mesh, ablate::mesh::BoxMesh, "a simple uniform box",
         ARG(std::string, "name", "the name of the mesh/domain"),
         ARG(std::map<std::string TMP_COMMA std::string>, "arguments", "arguments to be passed to petsc"),
         ARG(int, "dimensions", "the number of dimensions for the mesh"));
