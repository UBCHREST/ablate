#include "fieldDescription.hpp"
#include <map>
#include <regex>
#include <utility>
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscUtilities.hpp"

ablate::domain::FieldDescription::FieldDescription(std::string nameIn, const std::string& prefixIn, const std::vector<std::string>& componentsIn, ablate::domain::FieldLocation location,
                                                   ablate::domain::FieldType type, std::shared_ptr<domain::Region> region, const std::shared_ptr<parameters::Parameters>& optionsIn,
                                                   std::vector<std::string> tags)
    : name(std::move(nameIn)),
      prefix(prefixIn.empty() ? name + "_" : prefixIn + "_"),
      components(componentsIn.empty() ? std::vector<std::string>{"_"} : componentsIn),
      location(location),
      type(type),
      region(std::move(region)),
      tags(std::move(tags)),
      options(optionsIn),
      petscOptions(nullptr) {}

PetscObject ablate::domain::FieldDescription::CreatePetscField(DM dm) const {
    switch (type) {
        case FieldType::FEM: {
            // determine if it is a simplex element
            PetscBool simplex;
            DMPlexIsSimplex(dm, &simplex) >> utilities::PetscUtilities::checkError;
            PetscInt simplexLoc = simplex ? 1 : 0;
            PetscInt simplexGlobal;

            // Assume true if any rank says true
            MPI_Allreduce(&simplexLoc, &simplexGlobal, 1, MPIU_INT, MPIU_MAX, PetscObjectComm((PetscObject)dm)) >> ablate::utilities::MpiUtilities::checkError;

            // Determine the number of dims
            PetscInt dim;
            DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;

            // PetscFECreateDefault uses the petsc prefix to always search the global options.  If an options object was provided to this field, set it to global with the prefix
            if (options) {
                auto argumentMap = options->ToMap<std::string>();
                ablate::utilities::PetscUtilities::Set(prefix, argumentMap, false);
            }

            // create a petsc fe
            PetscFE petscFE;
            PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, (PetscInt)components.size(), simplexGlobal ? PETSC_TRUE : PETSC_FALSE, prefix.c_str(), PETSC_DEFAULT, &petscFE) >>
                utilities::PetscUtilities::checkError;
            PetscObjectSetName((PetscObject)petscFE, name.c_str()) >> utilities::PetscUtilities::checkError;

            // If this is not the first field, copy the quadrature locations
            // Check to see if there is already a petscFE object defined
            PetscInt numberFields;
            DMGetNumFields(dm, &numberFields) >> utilities::PetscUtilities::checkError;
            for (PetscInt f = 0; f < numberFields; f++) {
                PetscObject obj;
                DMGetField(dm, f, nullptr, &obj) >> utilities::PetscUtilities::checkError;
                PetscClassId id;
                PetscObjectGetClassId(obj, &id);

                if (id == PETSCFE_CLASSID) {
                    PetscFECopyQuadrature((PetscFE)obj, petscFE) >> utilities::PetscUtilities::checkError;
                }
            }
            return (PetscObject)petscFE;
        }
        case FieldType::FVM: {
            // If this is a fvm, create the petsc options from the parameters
            if (options) {
                PetscOptionsCreate(&petscOptions) >> utilities::PetscUtilities::checkError;
                options->Fill(petscOptions);
            }

            PetscFV fvm;
            PetscFVCreate(PetscObjectComm((PetscObject)dm), &fvm) >> utilities::PetscUtilities::checkError;
            PetscObjectSetName((PetscObject)fvm, name.c_str()) >> utilities::PetscUtilities::checkError;
            PetscObjectSetOptions((PetscObject)fvm, petscOptions) >> utilities::PetscUtilities::checkError;

            PetscFVSetFromOptions(fvm) >> utilities::PetscUtilities::checkError;
            PetscFVSetNumComponents(fvm, (PetscInt)components.size()) >> utilities::PetscUtilities::checkError;

            // Get the limiter
            PetscLimiter limiter;
            PetscFVGetLimiter(fvm, &limiter) >> utilities::PetscUtilities::checkError;
            PetscObjectSetOptions((PetscObject)limiter, petscOptions) >> utilities::PetscUtilities::checkError;
            PetscLimiterSetFromOptions(limiter) >> utilities::PetscUtilities::checkError;

            // Determine the number of dims
            PetscInt dim;
            DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;
            PetscFVSetSpatialDimension(fvm, dim) >> utilities::PetscUtilities::checkError;

            // Add the field to the
            return (PetscObject)fvm;
        }
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
ablate::domain::FieldDescription::~FieldDescription() {
    if (petscOptions) {
        ablate::utilities::PetscUtilities::PetscOptionsDestroyAndCheck("Field " + name, &petscOptions);
    }
}

#include "registrar.hpp"
REGISTER_DEFAULT_DERIVED(ablate::domain::FieldDescriptor, ablate::domain::FieldDescription);
REGISTER_DEFAULT(ablate::domain::FieldDescription, ablate::domain::FieldDescription, "A single custom field description", ARG(std::string, "name", "the name of the field"),
                 OPT(std::string, "prefix", "optional prefix (defaults to name)"), OPT(std::vector<std::string>, "components", "the components in the field (defaults to 1)"),
                 OPT(EnumWrapper<ablate::domain::FieldLocation>, "location", "if it is a solution (SOL) or auxiliary (aux) field"),
                 ARG(EnumWrapper<ablate::domain::FieldType>, "type", "if it is a finite volume (FV) or finite element (FE) field"),
                 OPT(ablate::domain::Region, "region", "the region in which this field lives"), OPT(ablate::parameters::Parameters, "options", "field specific options"),
                 OPT(std::vector<std::string>, "tags", "optional list of tags that can be used with this field"));
