#include "mesh.hpp"
#include "utilities/petscError.hpp"
int ablate::mesh::Mesh::GetDimensions() const {
    PetscInt dim;
    DMGetDimension(dm, &dim) >> checkError;
    return (int)dim;
}
