#ifndef ABLATELIBRARY_FIELDDESCRIPTOR_HPP
#define ABLATELIBRARY_FIELDDESCRIPTOR_HPP

#include <petsc.h>
#include <string>
#include <vector>
#include "field.hpp"
#include "parser/factory.hpp"

namespace ablate::domain {

struct FieldDescriptor {

    // Helper variable, replaces any components with this value with one for each dimension
    inline const static std::string DIMENSION = "_DIMENSION_";

    const std::string name;
    const std::string prefix;
    const std::vector<std::string> components = {"_"};
    const enum FieldType type = FieldType::SOL;
};


}  // namespace ablate::domain
#endif  // ABLATELIBRARY_FIELDDESCRIPTOR_HPP
