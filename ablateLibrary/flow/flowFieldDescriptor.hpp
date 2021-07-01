#ifndef ABLATELIBRARY_FLOWFIELDDESCRIPTOR_HPP
#define ABLATELIBRARY_FLOWFIELDDESCRIPTOR_HPP

#include <petsc.h>
#include <parser/factory.hpp>
#include <string>
#include <vector>
namespace ablate::flow {

enum class FieldType { FE, FV };

struct FlowFieldDescriptor {
    const bool solutionField = true;
    const std::string fieldName;
    const std::string fieldPrefix;
    const PetscInt components;
    const enum FieldType fieldType;
    const std::vector<std::string> componentNames;
};

std::istream& operator>>(std::istream& is, FieldType& v);

}  // namespace ablate::flow
#endif  // ABLATELIBRARY_FLOWFIELDDESCRIPTOR_HPP
