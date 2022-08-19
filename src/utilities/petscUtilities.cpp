#include "petscUtilities.hpp"
#include "environment/runEnvironment.hpp"
#include "petscError.hpp"

void ablate::utilities::PetscUtilities::Initialize(const char help[]) {
    PetscInitialize(ablate::environment::RunEnvironment::GetArgCount(), ablate::environment::RunEnvironment::GetArgs(), nullptr, help) >> checkError;

    // register the cleanup
    ablate::environment::RunEnvironment::RegisterCleanUpFunction("ablate::utilities::PetscUtilities::Initialize", []() { PetscFinalize() >> checkError; });
}
namespace ablate::utilities {

std::istream& operator>>(std::istream& is, PetscDataType& v) {
    // get the string
    std::string enumString;
    is >> enumString;

    // ask petsc for the enum
    PetscBool found;
    PetscDataTypeFromString(enumString.c_str(), &v, &found) >> checkError;

    if (!found) {
        v = PETSC_DATATYPE_UNKNOWN;
    }

    return is;
}

std::ostream& operator<<(std::ostream& os, const PetscDataType& type) {
    os << PetscDataTypes[type];
    return os;
}
}  // namespace ablate::utilities
