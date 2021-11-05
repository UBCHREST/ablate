#include "dmPlex.hpp"
#include <utilities/petscError.hpp>
#include <utilities/petscOptions.hpp>

ablate::domain::DMPlex::DMPlex(std::vector<std::shared_ptr<fields::FieldDescriptor>> fieldDescriptors, std::string nameIn, std::vector<std::shared_ptr<modifiers::Modifier>> modifiers)
    : Domain(CreateDM(nameIn), nameIn, fieldDescriptors, modifiers) {
}

ablate::domain::DMPlex::~DMPlex() {
    if (dm) {
        DMDestroy(&dm);
    }
}
DM ablate::domain::DMPlex::CreateDM(const std::string& name) {
    DM dm;
    DMCreate(PETSC_COMM_WORLD, &dm) >> checkError;
    DMSetType(dm, DMPLEX) >> checkError;
    PetscObjectSetName((PetscObject)dm, name.c_str()) >> checkError;
    return dm;
}

#include "parser/registrar.hpp"
REGISTER(ablate::domain::Domain, ablate::domain::DMPlex, "DMPlex that can be set using PETSc options",
         OPT(std::vector<domain::fields::FieldDescriptor>, "fields", "a list of fields/field descriptors"),
         ARG(std::string, "name", "the mesh dm name"),
         OPT(std::vector<domain::modifiers::Modifier>, "modifiers", "a list of domain modifier"));