#include "fieldDescription.hpp"
#include "utilities/petscUtilities.hpp"

ablate::particles::FieldDescription::FieldDescription(std::string name, ablate::domain::FieldLocation type, std::vector<std::string> componentsIn, PetscDataType dataTypeIn)
    : name(name), components(componentsIn.empty() ? std::vector<std::string>{"_"} : componentsIn), location(type), dataType(dataTypeIn == PETSC_DATATYPE_UNKNOWN ? PETSC_REAL : dataTypeIn) {}

using namespace ablate::utilities;
#include "registrar.hpp"
REGISTER_DEFAULT(ablate::particles::FieldDescription, ablate::particles::FieldDescription, "Describes a single field for particles", ARG(std::string, "name", "the name of the field"),
                 ARG(EnumWrapper<ablate::domain::FieldLocation>, "location", "if it is a solution (SOL) or auxiliary (aux) field"),
                 OPT(std::vector<std::string>, "components", "the components in this field. (default is a single component)"),
                 OPT(EnumWrapper<PetscDataType>, "dataType", "Possible PETSc data type (default is PETSC_REAL) "));
