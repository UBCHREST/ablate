#include "boxMesh.hpp"
#include "../utilities/petscError.hpp"

ablate::mesh::BoxMesh::BoxMesh(MPI_Comm comm, std::string name, std::map<std::string, std::string> arguments, int dimensions) : Mesh(comm, name, arguments) {
    DMPlexCreateBoxMesh(comm, dimensions, PETSC_TRUE, NULL, NULL, NULL, NULL, PETSC_TRUE, &dm) >> checkError;
    DMSetOptionsPrefix(dm, name.c_str()) >> checkError;
    DMSetFromOptions(dm) >> checkError;
    DMViewFromOptions(dm, NULL, "-dm_view") >> checkError;
}
