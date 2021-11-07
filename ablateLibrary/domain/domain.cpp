#include "domain.hpp"
#include <utilities/mpiError.hpp>
#include "solver/solver.hpp"
#include "subDomain.hpp"
#include "utilities/petscError.hpp"

ablate::domain::Domain::Domain(DM dmIn, std::string name, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptorsIn, std::vector<std::shared_ptr<modifiers::Modifier>> modifiersIn)
    : dm(dmIn), name(name), comm(PetscObjectComm((PetscObject)dm)), fieldDescriptors(std::move(fieldDescriptorsIn)), solField(nullptr), modifiers(std::move(modifiersIn)) {
    // sort the modifiers based upon priority
    std::sort(modifiers.begin(), modifiers.end(), [](const auto& a, const auto& b) -> bool { return a->Priority() < b->Priority(); });

    // update the dm with the modifiers
    for (auto& modifier : modifiers) {
        modifier->Modify(dm);
    }

    // register all the solution fields with the DM, store the aux fields for later
    std::vector<std::shared_ptr<FieldDescription>> allAuxFields;
    for (const auto& fieldDescriptor : fieldDescriptors) {
        for (auto& fieldDescription : fieldDescriptor->GetFields()) {
            fieldDescription->DecompressComponents(GetDimensions());
            switch (fieldDescription->location) {
                case FieldLocation::SOL:
                    RegisterField(*fieldDescription);
                    break;
                case FieldLocation::AUX:
                    allAuxFields.push_back(fieldDescription);
                    break;
                default:
                    throw std::invalid_argument("Unknown Field Location for " + fieldDescription->name);
            }
        }
    }

    // Set up the global DS
    DMCreateDS(dm) >> checkError;

    // based upon the ds divisions in the dm, create a subDomain for each
    PetscInt numberDS;
    DMGetNumDS(dm, &numberDS) >> checkError;

    // March over each ds and create a subDomain
    for (PetscInt ds = 0; ds < numberDS; ds++) {
        subDomains.emplace_back(std::make_shared<ablate::domain::SubDomain>(*this, ds, allAuxFields));
    }
}

ablate::domain::Domain::~Domain() {
    // clean up the petsc objects
    if (solField) {
        VecDestroy(&solField) >> checkError;
    }
}

void ablate::domain::Domain::RegisterField(const ablate::domain::FieldDescription& fieldDescription) {
    // make sure that this is a solution field
    if (fieldDescription.location != FieldLocation::SOL) {
        throw std::invalid_argument("The field must be FieldLocation::SOL to be registered with the domain");
    }

    // Look up the label for this field
    DMLabel label = nullptr;
    if (fieldDescription.region) {
        DMGetLabel(dm, fieldDescription.region->GetName().c_str(), &label) >> checkError;
        if (label == nullptr) {
            throw std::invalid_argument("Cannot locate label " + fieldDescription.region->GetName() + " for field " + fieldDescription.name);
        }
    }

    // Create the field and add it with the label
    auto petscField = fieldDescription.CreatePetscField(dm);

    // add to the dm
    DMAddField(dm, label, petscField);

    // Free the petsc after being added
    PetscObjectDestroy(&petscField);

    // Record the field
    fields.push_back(Field::FromFieldDescription(fieldDescription, fields.size()));
}

PetscInt ablate::domain::Domain::GetDimensions() const {
    PetscInt dim;
    DMGetDimension(dm, &dim) >> checkError;
    return dim;
}

void ablate::domain::Domain::CreateStructures() {
    // Setup the solve with the ts
    DMPlexCreateClosureIndex(dm, NULL) >> checkError;
    DMCreateGlobalVector(dm, &(solField)) >> checkError;
    PetscObjectSetName((PetscObject)solField, "solution") >> checkError;
}

std::shared_ptr<ablate::domain::SubDomain> ablate::domain::Domain::GetSubDomain(std::shared_ptr<domain::Region> region) {
    // Check to see if there is a label for this region
    if (region) {
        // March over each ds region, and return the subdomain if this region is inside of any subDomain region
        for (const auto& subDomain : subDomains) {
            if (subDomain->InRegion(*region)) {
                return subDomain;
            }
        }
        throw std::runtime_error("Unable to locate subDomain for region " + region->ToString());

    } else {
        // Get the only subDomain
        if (subDomains.size() > 1) {
            throw std::runtime_error("More than one DS was created, the region is expected to be defined.");
        }
        return subDomains.front();
    }
}

void ablate::domain::Domain::InitializeSubDomains(std::vector<std::shared_ptr<solver::Solver>> solvers) {
    // determine the number of fields
    for (auto& solver : solvers) {
        solver->Register(GetSubDomain(solver->GetRegion()));
    }

    // Setup each of the fields
    for (auto& solver : solvers) {
        solver->Setup();
    }
    // Create the global structures
    CreateStructures();
    for (auto& subDomain : subDomains) {
        subDomain->CreateSubDomainStructures();
    }

    // Initialize each solver
    for (auto& solver : solvers) {
        solver->Initialize();
    }
}
