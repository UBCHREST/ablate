#ifndef ABLATELIBRARY_FIELDDESCRIPTOR_HPP
#define ABLATELIBRARY_FIELDDESCRIPTOR_HPP

#include <petsc.h>
#include <string>
#include <vector>
#include "field.hpp"
#include "parser/factory.hpp"

namespace ablate::domain {

struct FieldDescriptor {
    const std::string fieldName;
    const std::string fieldPrefix;
    const PetscInt components;
    const std::vector<std::string> componentNames;
    const enum FieldLocation fieldLocation = FieldLocation::SOL;
};


}  // namespace ablate::domain
#endif  // ABLATELIBRARY_FIELDDESCRIPTOR_HPP
