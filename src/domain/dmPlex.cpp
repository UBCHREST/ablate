#include "dmPlex.hpp"
#include <utility>
#include "utilities/petscUtilities.hpp"

ablate::domain::DMPlex::DMPlex(std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors, const std::string& nameIn, std::vector<std::shared_ptr<modifiers::Modifier>> modifiers,
                               std::shared_ptr<parameters::Parameters> options)
    : Domain(CreateDM(nameIn), nameIn, std::move(fieldDescriptors), std::move(modifiers), std::move(options)) {}

ablate::domain::DMPlex::~DMPlex() {
    if (dm) {
        DMDestroy(&dm);
    }
}
DM ablate::domain::DMPlex::CreateDM(const std::string& name) {
    DM dm;
    DMCreate(PETSC_COMM_WORLD, &dm) >> utilities::PetscUtilities::checkError;
    DMSetType(dm, DMPLEX) >> utilities::PetscUtilities::checkError;
    PetscObjectSetName((PetscObject)dm, name.c_str()) >> utilities::PetscUtilities::checkError;
    return dm;
}

#include "registrar.hpp"
REGISTER(ablate::domain::Domain, ablate::domain::DMPlex, "DMPlex that can be set using PETSc options",
         OPT(std::vector<ablate::domain::FieldDescriptor>, "fields", "a list of fields/field descriptors"), ARG(std::string, "name", "the mesh dm name"),
         OPT(std::vector<ablate::domain::modifiers::Modifier>, "modifiers", "a list of domain modifier"),
         OPT(ablate::parameters::Parameters, "options", "PETSc options specific to this dm.  Default value allows the dm to access global options."));