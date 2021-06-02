#include "mesh.hpp"
#include "../utilities/petscOptions.hpp"

ablate::mesh::Mesh::Mesh(std::string name, std::map<std::string, std::string> arguments) : name(name), dm(nullptr) {
    // append any prefix values
    utilities::PetscOptionsUtils::Set(name, arguments);
}