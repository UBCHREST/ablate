#include "boxMesh.hpp"
#include <utilities/mpiError.hpp>
#include "../utilities/petscError.hpp"
#include "parser/registrar.hpp"

ablate::mesh::BoxMesh::BoxMesh(std::string name, std::map<std::string, std::string> arguments, std::vector<int> faces, std::vector<double> lower, std::vector<double> upper)
    : Mesh(PETSC_COMM_WORLD, name, Merge(arguments, {{"dm_distribute", "true"}})) {
    PetscInt dimensions = faces.size();
    if ((dimensions != lower.size()) || (dimensions != upper.size())) {
        throw std::runtime_error("BoxMesh Error: The faces, lower, and upper vectors must all be the same dimension.");
    }

    DMPlexCreateBoxMesh(comm, dimensions, PETSC_TRUE, &faces[0], &lower[0], &upper[0], NULL, PETSC_TRUE, &dm) >> checkError;
    DMSetOptionsPrefix(dm, name.c_str()) >> checkError;
    DMSetFromOptions(dm) >> checkError;

    IS globalCellNumbers;
    DMPlexGetCellNumbering(dm, &globalCellNumbers) >> checkError;
    PetscInt size;
    ISGetLocalSize(globalCellNumbers, &size) >> checkError;
    if (size == 0) {
        int rank;
        MPI_Comm_rank(comm, &rank) >> checkMpiError;
        throw std::runtime_error("BoxMesh Error: Rank " + std::to_string(rank) + " distribution resulted in no cells.  Increase the number of cells in each direction.");
    }
}

REGISTER(ablate::mesh::Mesh, ablate::mesh::BoxMesh, "a simple uniform box", ARG(std::string, "name", "the name of the mesh/domain"),
         ARG(std::map<std::string TMP_COMMA std::string>, "arguments", "arguments to be passed to petsc"), ARG(std::vector<int>, "faces", "the number of faces in each direction for the mesh"),
         ARG(std::vector<double>, "lower", "the lower bound for the mesh"), ARG(std::vector<double>, "upper", "the upper bound for the mesh"));
