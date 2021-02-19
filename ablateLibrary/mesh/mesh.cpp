#include "mesh.hpp"
#include "../utilities/petscOptions.hpp"

ablate::mesh::Mesh::Mesh(MPI_Comm comm, std::string name, std::map<std::string, std::string> arguments) : name(name), dm(nullptr), comm(comm) {
    // append any prefix values
    utilities::PetscOptions::Set(name, arguments);
}

ablate::mesh::Mesh::~Mesh() {
    if (dm) {
        DMDestroy(&dm);
    }
}
