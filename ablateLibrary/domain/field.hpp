#ifndef ABLATELIBRARY_FIELD_HPP
#define ABLATELIBRARY_FIELD_HPP

#include <petsc.h>
#include <string>
#include <vector>

namespace ablate::domain {

enum class FieldLocation { SOL, AUX };

struct Field {
    std::string fieldName;
    PetscInt components;
    std::vector<std::string> componentNames;
    PetscInt fieldId;
    enum FieldLocation fieldLocation = FieldLocation::SOL;
};

std::istream& operator>>(std::istream& is, FieldLocation& v);

}  // namespace ablate::domain
#endif  // ABLATELIBRARY_FIELD_HPP
