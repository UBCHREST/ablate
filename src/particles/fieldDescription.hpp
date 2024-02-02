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
    enum domain::FieldLocation location = domain::FieldLocation::AUX;

    //! The data type, default is double
    const PetscDataType dataType = PETSC_REAL;

   public:
    /**
     * default constructor
     * @param name name of the field
     * @param type AUX or SOL type
     * @param components the name of each components, if empty it it is assumed to be one component
     * @param dataType the type of data, default is real
     */
    FieldDescription(std::string name, domain::FieldLocation type, const std::vector<std::string>& components = {}, PetscDataType dataType = PETSC_DATATYPE_UNKNOWN);
};

}  // namespace ablate::particles
#endif  // ABLATELIBRARY_PARTICLEFIELDDESCRIPTION_HPP
