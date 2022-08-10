#ifndef ABLATELIBRARY_PARTICLEFIELD_HPP
#define ABLATELIBRARY_PARTICLEFIELD_HPP

#include <petsc.h>
#include <string>
#include <vector>
#include "domain/field.hpp"

namespace ablate::particles {

struct Field {
    //! The unique name of the particle field
    const std::string name;

    //! The number of the components
    const PetscInt numberComponents;

    //! The name of the components
    const std::vector<std::string> components;

    //! The field type (sol or aux)
    enum domain::FieldLocation type = domain::FieldLocation::AUX;

    //! The type of field
    const PetscDataType dataType;

    //! The offset in the local array, 0 for aux, computed for sol
    const PetscInt offset = 0;

    //! The size of the component for this data
    PetscInt dataSize = 0;

    //! Inline function to compute offset
    template <class IndexType>
    inline IndexType operator[](IndexType particle) const{
        return particle*dataSize + offset;
    }
};

}  // namespace ablate::particles
#endif  // ABLATELIBRARY_FIELD_HPP
