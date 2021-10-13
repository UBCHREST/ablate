#include "domain.hpp"
#include "utilities/petscError.hpp"
int ablate::domain::Domain::GetDimensions() const {
    PetscInt dim;
    DMGetDimension(dm, &dim) >> checkError;
    return (int)dim;
}
