#ifndef ABLATELIBRARY_FLOWFIELDDESCRIPTOR_HPP
#define ABLATELIBRARY_FLOWFIELDDESCRIPTOR_HPP

#include <petsc.h>
#include <string>
#include <vector>
namespace ablate::flow {

enum class FieldType { FE, FV };

struct FlowFieldDescriptor {
    const std::string fieldName;
    const std::string fieldPrefix;
    const PetscInt components;
    const enum FieldType fieldType;
    const std::vector<std::string> componentNames;
};
}  // namespace ablate::flow
#endif  // ABLATELIBRARY_FLOWFIELDDESCRIPTOR_HPP
