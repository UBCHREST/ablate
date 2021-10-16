#ifndef ABLATELIBRARY_FIELD_HPP
#define ABLATELIBRARY_FIELD_HPP

#include <petsc.h>
#include <string>
#include <vector>

namespace ablate::domain {

enum class FieldType { SOL, AUX };

struct Field {
    std::string name;
    PetscInt numberComponents;
    std::vector<std::string> components;
    PetscInt id;
    enum FieldType type = FieldType::SOL;
};

std::istream& operator>>(std::istream& is, FieldType& v);

}  // namespace ablate::domain
#endif  // ABLATELIBRARY_FIELD_HPP
