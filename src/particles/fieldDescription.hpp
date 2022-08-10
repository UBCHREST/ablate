#ifndef ABLATELIBRARY_PARTICLEFIELDDESCRIPTION_HPP
#define ABLATELIBRARY_PARTICLEFIELDDESCRIPTION_HPP

#include <petsc.h>
#include <string>
#include <vector>
#include "domain/field.hpp"

namespace ablate::particles {

struct FieldDescription {
    //! The unique name of the particle field
    const std::string name;

    //! Optional name of the components.  This is used to determine the number of components
    const std::vector<std::string> components = {"_"};

    //! The type of field, (solution or aux)
    enum domain::FieldLocation type = domain::FieldLocation::AUX;

    //! The data type, default is double
    const PetscDataType dataType = PETSC_REAL;
};

}  // namespace ablate::particles
#endif  // ABLATELIBRARY_FIELD_HPP
