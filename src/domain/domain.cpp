#include "domain.hpp"
#include <typeinfo>
#include <utilities/mpiError.hpp>
#include <utility>
#include "monitors/logs/stdOut.hpp"
#include "solver/solver.hpp"
#include "subDomain.hpp"
#include "utilities/demangler.hpp"
#include "utilities/petscError.hpp"
#include "utilities/petscOptions.hpp"

ablate::domain::Domain::Domain(DM dmIn, std::string name, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptorsIn, std::vector<std::shared_ptr<modifiers::Modifier>> modifiersIn,
                               const std::shared_ptr<parameters::Parameters>& options)
    : dm(dmIn), name(std::move(name)), comm(PetscObjectComm((PetscObject)dm)), fieldDescriptors(std::move(fieldDescriptorsIn)), solGlobalField(nullptr), modifiers(std::move(modifiersIn)) {
    // if provided, convert options to a petscOptions
    if (options) {
        PetscOptionsCreate(&petscOptions) >> checkError;
        options->Fill(petscOptions);
    }

    // Apply petsc options to the domain
    PetscObjectSetOptions((PetscObject)dm, petscOptions) >> checkError;
    DMSetFromOptions(dm) >> checkError;

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
    if (solGlobalField) {
        VecDestroy(&solGlobalField) >> checkError;
    }
    if (petscOptions) {
        ablate::utilities::PetscOptionsDestroyAndCheck("ablate::domain::Domain", &petscOptions);
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
    fields.push_back(Field::FromFieldDescription(fieldDescription, (PetscInt)fields.size()));
}

PetscInt ablate::domain::Domain::GetDimensions() const noexcept {
    PetscInt dim;
    DMGetDimension(dm, &dim) >> checkError;
    return dim;
}

void ablate::domain::Domain::CreateStructures() {
    // Setup the solve with the ts
    DMPlexCreateClosureIndex(dm, nullptr) >> checkError;
    DMCreateGlobalVector(dm, &(solGlobalField)) >> checkError;
    PetscObjectSetName((PetscObject)solGlobalField, "solution") >> checkError;

    // add the names to each of the components in the dm section
    PetscSection section;
    DMGetLocalSection(dm, &section) >> checkError;
    for (const auto& field : fields) {
        if (field.numberComponents > 1) {
            for (PetscInt c = 0; c < field.numberComponents; c++) {
                PetscSectionSetComponentName(section, field.id, c, field.components[c].c_str()) >> checkError;
            }
        }
    }
}

std::shared_ptr<ablate::domain::SubDomain> ablate::domain::Domain::GetSubDomain(const std::shared_ptr<domain::Region>& region) {
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

void ablate::domain::Domain::InitializeSubDomains(const std::vector<std::shared_ptr<solver::Solver>>& solvers, const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& initializations,
                                                  const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& exactSolutions) {
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

    // set all values to nan to allow for a output check
    if (!initializations.empty()) {
        VecSet(solGlobalField, NAN) >> checkError;
    }

    // Set the initial conditions for each field specified
    ProjectFieldFunctions(initializations, solGlobalField);

    // do a sanity check to make sure that all points were initialized
    if (!initializations.empty() && CheckSolution()) {
        throw std::runtime_error("Field values at points in the domain were not initialized.");
    }

    // Initialize each solver
    for (auto& solver : solvers) {
        solver->Initialize();
    }

    // Set the exact solutions if the field in lives in each subDomain
    for (auto& subDomain : subDomains) {
        subDomain->SetsExactSolutions(exactSolutions);
    }
}
std::vector<std::weak_ptr<ablate::io::Serializable>> ablate::domain::Domain::GetSerializableSubDomains() {
    std::vector<std::weak_ptr<io::Serializable>> serializables;
    for (auto& serializable : subDomains) {
        serializables.push_back(serializable);
    }
    return serializables;
}

void ablate::domain::Domain::ProjectFieldFunctions(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& fieldFunctions, Vec globVec, PetscReal time) {
    PetscInt numberFields;
    DMGetNumFields(dm, &numberFields) >> checkError;

    // get a local vector for the work
    Vec locVec;
    DMGetLocalVector(dm, &locVec) >> checkError;
    DMGlobalToLocal(dm, globVec, INSERT_VALUES, locVec) >> checkError;

    for (auto& fieldFunction : fieldFunctions) {
        // Size up the field projects
        std::vector<mathFunctions::PetscFunction> fieldFunctionsPts(numberFields, nullptr);
        std::vector<void*> fieldContexts(numberFields, nullptr);

        auto fieldId = GetField(fieldFunction->GetName());
        fieldContexts[fieldId.id] = fieldFunction->GetSolutionField().GetContext();
        fieldFunctionsPts[fieldId.id] = fieldFunction->GetSolutionField().GetPetscFunction();

        // Determine where to apply this field
        DMLabel fieldLabel = nullptr;
        PetscInt fieldValue = 0;
        if (const auto& region = fieldFunction->GetRegion()) {
            fieldValue = region->GetValue();
            DMGetLabel(dm, region->GetName().c_str(), &fieldLabel) >> checkError;
        } else {
            PetscObject fieldTemp;
            DMGetField(dm, fieldId.id, &fieldLabel, &fieldTemp) >> checkError;
            if (fieldLabel) {
                fieldValue = 1;  // this is temporary until petsc allows fields to be defined with values beside 1
            }
        }

        // Note the global DMProjectFunctionLabel can't be used because it overwrites unwritten values.
        // Project this field
        if (fieldLabel) {
            // make sure that some of this field exists here
            IS regionIS;
            DMLabelGetStratumIS(fieldLabel, fieldValue, &regionIS) >> checkError;

            if (regionIS) {
                DMProjectFunctionLabelLocal(dm, time, fieldLabel, 1, &fieldValue, -1, nullptr, fieldFunctionsPts.data(), fieldContexts.data(), INSERT_VALUES, locVec) >> checkError;
                ISDestroy(&regionIS) >> checkError;
            }
        } else {
            DMProjectFunctionLocal(dm, time, fieldFunctionsPts.data(), fieldContexts.data(), INSERT_VALUES, locVec) >> checkError;
        }
    }

    // push the results back to the global vector
    DMLocalToGlobal(dm, locVec, INSERT_VALUES, globVec) >> checkError;
    DMRestoreLocalVector(dm, &locVec) >> checkError;
}

bool ablate::domain::Domain::CheckSolution() {
    bool error = false;
    for (auto& subdomain : subDomains) {
        error = subdomain->CheckSolution() | error;
    }
    return error;
}
