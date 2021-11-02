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

    std::string name;
    std::string prefix;
    std::vector<std::string> components = {"_"};
    enum FieldType type = FieldType::SOL;
    enum FieldAdjacency adjacency = FieldAdjacency::DEFAULT;
};

}  // namespace ablate::domain
#endif  // ABLATELIBRARY_FIELDDESCRIPTOR_HPP
