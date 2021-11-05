#include "fieldDescription.hpp"
#include <map>
#include <utilities/mpiError.hpp>
#include <utilities/petscError.hpp>
#include <regex>

ablate::domain::FieldDescription::FieldDescription(std::string nameIn, std::string prefixIn, std::vector<std::string> componentsIn, ablate::domain::FieldLocation location,
                                                           ablate::domain::FieldType type, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options)
    : name(nameIn),
      prefix(prefixIn.empty() ? name : prefixIn),
      components(componentsIn.empty() ? std::vector<std::string>{"_"} : componentsIn),
      location(location),
      type(type),
      region(region),
      options(options) {}


PetscObject ablate::domain::FieldDescription::CreatePetscField(DM dm) const {
    switch (type) {
        case FieldType::FEM: {
            // determine if it a simplex element and the number of dimensions
            DMPolytopeType ct;
            PetscInt cStart;
            DMPlexGetHeightStratum(dm, 0, &cStart, NULL) >> checkError;
            DMPlexGetCellType(dm, cStart, &ct) >> checkError;
            PetscInt simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1 ? PETSC_TRUE : PETSC_FALSE;
            PetscInt simplexGlobal;

            // Assume true if any rank says true
            MPI_Allreduce(&simplex, &simplexGlobal, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm)) >> checkMpiError;

            // Determine the number of dims
            PetscInt dim;
            DMGetDimension(dm, &dim) >> checkError;

            // create a petsc fe
            PetscFE petscFE;
            PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, components.size(), simplexGlobal ? PETSC_TRUE : PETSC_FALSE, prefix.c_str(), PETSC_DEFAULT, &petscFE) >>
                checkError;
            PetscObjectSetName((PetscObject)petscFE, name.c_str()) >> checkError;
            // TODO: move this to specific options
            PetscObjectSetOptions((PetscObject)petscFE, nullptr) >> checkError;

            // If this is not the first field, copy the quadrature locations
            // Check to see if there is already a petscFE object defined
            PetscInt numberFields;
            DMGetNumFields(dm, &numberFields) >> checkError;
            for(PetscInt f = 0; f < numberFields; f++){
                PetscObject obj;
                DMGetField(dm, f, NULL, &obj) >> checkError;
                PetscClassId id;
                PetscObjectGetClassId(obj, &id);

                if (id == PETSCFE_CLASSID) {
                    PetscFECopyQuadrature((PetscFE)obj, petscFE) >> checkError;
                }
            }
            return (PetscObject)petscFE;
        }
        case FieldType::FVM: {
            PetscFV fvm;
            PetscFVCreate(PetscObjectComm((PetscObject)dm), &fvm) >> checkError;
            PetscObjectSetOptionsPrefix((PetscObject)fvm, prefix.c_str()) >> checkError;
            PetscObjectSetName((PetscObject)fvm, name.c_str()) >> checkError;

            // TODO: convert to petscOptions
            // PetscObjectSetOptions((PetscObject)fvm, petscOptions) >> checkError;

            // Determine the number of dims
            PetscInt dim;
            DMGetDimension(dm, &dim) >> checkError;

            PetscFVSetFromOptions(fvm) >> checkError;
            PetscFVSetNumComponents(fvm, components.size()) >> checkError;
            PetscFVSetSpatialDimension(fvm,dim) >> checkError;

            // Add the field to the
            return (PetscObject)fvm;
        } break;
        default:
            throw std::runtime_error("Can only register SOL fields in Domain::RegisterSolutionField");
    }
}
void ablate::domain::FieldDescription::DecompressComponents(PetscInt ndims) {
    for (std::size_t c = 0; c < components.size(); c++) {
        if (components[c].find(FieldDescription::DIMENSION) != std::string::npos) {
            auto baseName = components[c];

            // Delete this component
            components.erase(components.begin() + c);

            for (PetscInt d = ndims - 1; d >= 0; d--) {
                auto newName = std::regex_replace(baseName, std::regex(FieldDescription::DIMENSION), std::to_string(d));
                components.insert(components.begin() + c, newName);
            }
        }
    }
}
std::vector<std::shared_ptr<ablate::domain::FieldDescription>> ablate::domain::FieldDescription::GetFields() {
    return std::vector<std::shared_ptr<ablate::domain::FieldDescription>>{shared_from_this()};
}

#include "parser/registrar.hpp"
REGISTER(ablate::domain::FieldDescription, ablate::domain::FieldDescription, "A single custom field description",
         ARG(std::string, "name", "the name of the field"),
         OPT(std::string, "prefix", "optional prefix (defaults to name)"),
         OPT(std::vector<std::string>, "components", "the components in the field (defaults to 1)"),
         OPT(EnumWrapper<ablate::domain::FieldLocation>, "location", "if it is a solution (SOL) or auxiliary (aux) field"),
         ARG(EnumWrapper<ablate::domain::FieldType>, "type", "if it is a finite volume (FV) or finite element (FE) field"),
         OPT(domain::Region, "region", "the region in which this field lives"),
         OPT(parameters::Parameters, "options", "field specific options"));

REGISTER(ablate::domain::FieldDescriptor, ablate::domain::FieldDescription, "A single custom field description",
         ARG(std::string, "name", "the name of the field"),
         OPT(std::string, "prefix", "optional prefix (defaults to name)"),
         OPT(std::vector<std::string>, "components", "the components in the field (defaults to 1)"),
         OPT(EnumWrapper<ablate::domain::FieldLocation>, "location", "if it is a solution (SOL) or auxiliary (aux) field"),
         ARG(EnumWrapper<ablate::domain::FieldType>, "type", "if it is a finite volume (FV) or finite element (FE) field"),
         OPT(domain::Region, "region", "the region in which this field lives"),
         OPT(parameters::Parameters, "options", "field specific options"));