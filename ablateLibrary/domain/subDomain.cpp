#include "subDomain.hpp"
#include <utilities/petscError.hpp>

ablate::domain::SubDomain::SubDomain(std::weak_ptr<Domain> domain, std::shared_ptr<domain::Region> region)
    : domain(domain), region(region), name(""), label(nullptr), auxDM(nullptr), auxVec(nullptr), subDM(nullptr), subSolutionVec(nullptr), subAuxDM(nullptr), subAuxVec(nullptr) {
    if (region) {
        if (auto domainPtr = domain.lock()) {
            DMGetLabel(domainPtr->GetDM(), region->GetName().c_str(), &label) >> checkError;
        } else {
            throw std::runtime_error("Cannot Locate Field in DM. DM is NULL");
        }

        name = region->GetName();
    }
}

ablate::domain::SubDomain::~SubDomain() {
    if (auxDM) {
        DMDestroy(&auxDM) >> checkError;
    }

    if (auxVec) {
        VecDestroy(&auxVec) >> checkError;
    }

    if (subDM) {
        DMDestroy(&subDM) >> checkError;
    }

    if (subSolutionVec) {
        VecDestroy(&subSolutionVec) >> checkError;
    }

    if (subAuxDM) {
        DMDestroy(&subAuxDM) >> checkError;
    }

    if (subAuxVec) {
        VecDestroy(&subAuxVec) >> checkError;
    }
}

ablate::domain::Field ablate::domain::SubDomain::RegisterField(const ablate::domain::FieldDescriptor& fieldDescriptor, PetscObject field) {
    // Create a field with this information
    Field newField{.name = fieldDescriptor.name, .numberComponents = (PetscInt)fieldDescriptor.components.size(), .components = fieldDescriptor.components, .id = -1, .type = fieldDescriptor.type};

    // Store the location in this subdomain
    newField.id = fieldsByType[fieldDescriptor.type].size();

    // store by name
    fieldsByName[newField.name] = newField;
    // also store the values by type
    fieldsByType[fieldDescriptor.type].push_back(newField);

    if (auto domainPtr = domain.lock()) {
        // add solution fields/aux fields
        switch (fieldDescriptor.type) {
            case FieldType::SOL: {
                domainPtr->RegisterSolutionField(fieldDescriptor, field, label);
                break;
            }
            case FieldType::AUX: {
                // check to see if need to create an aux dm
                if (auxDM == nullptr) {
                    /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
                    DM coordDM;
                    DMGetCoordinateDM(domainPtr->GetDM(), &coordDM) >> checkError;
                    DMClone(domainPtr->GetDM(), &auxDM) >> checkError;

                    // this is a hard coded "dmAux" that petsc looks for
                    DMSetCoordinateDM(auxDM, coordDM) >> checkError;
                }
                DMAddField(auxDM, label, (PetscObject)field) >> checkError;
            }
        }

    } else {
        throw std::runtime_error("Cannot RegisterField " + newField.name + ". Domain is expired.");
    }

    return newField;
}

DM& ablate::domain::SubDomain::GetDM() {
    if (auto domainPtr = domain.lock()) {
        return domainPtr->GetDM();
    } else {
        throw std::runtime_error("Cannot Get DM. Domain is expired.");
    }
}

DM ablate::domain::SubDomain::GetAuxDM() { return auxDM; }

Vec ablate::domain::SubDomain::GetSolutionVector() {
    if (auto domainPtr = domain.lock()) {
        return domainPtr->GetSolutionVector();
    } else {
        throw std::runtime_error("Cannot Get DM. Domain is expired.");
    }
}

Vec ablate::domain::SubDomain::GetAuxVector() { return auxVec; }

PetscInt ablate::domain::SubDomain::GetDimensions() const {
    if (auto domainPtr = domain.lock()) {
        return domainPtr->GetDimensions();
    } else {
        throw std::runtime_error("Cannot Get DM. Domain is expired.");
    }
}

void ablate::domain::SubDomain::CreateSubDomainStructures() {
    if (auto domainPtr = domain.lock()) {
        if (auxDM) {
            DMCreateDS(auxDM) >> checkError;
            DMCreateLocalVector(auxDM, &(auxVec)) >> checkError;

            DMGetRegionDS(auxDM, label, nullptr, &auxDiscreteSystem) >> checkError;

            // attach this field as aux vector to the dm
            DMSetAuxiliaryVec(domainPtr->GetDM(), label, label ? region->GetValues()[0] : 0, auxVec) >> checkError;
            auto vecName = "aux" + (region ? "_" + region->GetName() : "");
            PetscObjectSetName((PetscObject)auxVec, vecName.c_str()) >> checkError;
        }
    } else {
        throw std::runtime_error("Cannot CreateSubDomainStructures. Domain is expired.");
    }
}

void ablate::domain::SubDomain::InitializeDiscreteSystem() {
    if (auto domainPtr = domain.lock()) {
        DMGetRegionDS(domainPtr->GetDM(), label, nullptr, &discreteSystem) >> checkError;
    } else {
        throw std::runtime_error("Cannot CreateSubDomainStructures. Domain is expired.");
    }
}

PetscObject ablate::domain::SubDomain::GetPetscFieldObject(const Field& field) {
    switch (field.type) {
        case FieldType::SOL: {
            auto solutionField = GetSolutionField(field.name);
            PetscObject fieldObject;
            DMGetField(GetDM(), solutionField.id, nullptr, &fieldObject) >> checkError;
            return fieldObject;
        }
        case FieldType::AUX: {
            PetscObject fieldObject;
            DMGetField(auxDM, field.id, nullptr, &fieldObject) >> checkError;
            return fieldObject;
        }
    }
    return nullptr;
}

void ablate::domain::SubDomain::ProjectFieldFunctions(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& initialization, Vec globVec, PetscReal time) {
    PetscInt numberFields;
    DM dm;

    VecGetDM(globVec, &dm) >> checkError;
    DMGetNumFields(dm, &numberFields) >> checkError;

    // size up the update and context functions
    std::vector<mathFunctions::PetscFunction> fieldFunctions(numberFields, NULL);
    std::vector<void*> fieldContexts(numberFields, NULL);

    for (auto fieldInitialization : initialization) {
        auto fieldId = GetSolutionField(fieldInitialization->GetName());

        fieldContexts[fieldId.id] = fieldInitialization->GetSolutionField().GetContext();
        fieldFunctions[fieldId.id] = fieldInitialization->GetSolutionField().GetPetscFunction();
    }

    DMProjectFunction(dm, time, &fieldFunctions[0], &fieldContexts[0], INSERT_VALUES, globVec) >> checkError;
}

DM ablate::domain::SubDomain::GetSubDM() {
    // If there is no label, just return the entire dm
    if (!label) {
        return GetDM();
    }

    // If the subDM has not been created, create one
    if (!subDM) {
        // filter by label
        DMPlexFilter(GetDM(), label, region->GetValues().front(), &subDM) >> checkError;

        // copy over all fields that were in the main dm
        for (auto& fieldInfo : GetFields()) {
            auto petscField = GetPetscFieldObject(fieldInfo);
            DMAddField(subDM, NULL, petscField) >> checkError;
        }
    }

    return subDM;
}

Vec ablate::domain::SubDomain::GetSubSolutionVector() {
    // If there is no label, just return the entire solution vector
    if (!label) {
        return GetSolutionVector();
    }

    GetSubDM();
    if (!subSolutionVec) {
        DMCreateGlobalVector(subDM, &subSolutionVec) >> checkError;

        // Assign a name the same as the global
        const char* vecName;
        PetscObjectGetName((PetscObject)GetSolutionVector(), &vecName) >> checkError;
        PetscObjectSetName((PetscObject)subSolutionVec, vecName) >> checkError;
    }

    CopyGlobalToSubVector(subDM, GetDM(), subSolutionVec, GetSolutionVector(), GetFields());

    return subSolutionVec;
}

DM ablate::domain::SubDomain::GetSubAuxDM() {
    // If there is no auxDM, there cannot be a sub Aux DM
    if (!auxDM) {
        return nullptr;
    }

    // If there is no label, just return the entire dm
    if (!label) {
        return GetAuxDM();
    }

    // If it is already created, return it
    if (subAuxDM) {
        return subAuxDM;
    }

    // Create a sub auxDM
    /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
    DM coordDM;
    DMGetCoordinateDM(GetSubDM(), &coordDM) >> checkError;
    DMClone(GetSubDM(), &subAuxDM) >> checkError;

    // this is a hard coded "dmAux" that petsc looks for
    DMSetCoordinateDM(subAuxDM, coordDM) >> checkError;

    // Add all of the fields
    // copy over all fields that were in the main dm
    for (auto& fieldInfo : GetFields(FieldType::AUX)) {
        auto petscField = GetPetscFieldObject(fieldInfo);
        DMAddField(subAuxDM, NULL, petscField) >> checkError;
    }

    return subAuxDM;
}

Vec ablate::domain::SubDomain::GetSubAuxVector() {
    // If there is no auxVector, return null
    if (!auxVec) {
        return nullptr;
    }

    // If there is no label, just return the entire solution vector
    if (!label) {
        return GetAuxVector();
    }

    if (!subAuxVec) {
        DMCreateGlobalVector(GetSubAuxDM(), &subAuxVec) >> checkError;
    }

    CopyGlobalToSubVector(GetSubAuxDM(), GetAuxDM(), subAuxVec, GetAuxVector(), GetFields(FieldType::AUX), GetFields(FieldType::AUX), true);
    return subAuxVec;
}

void ablate::domain::SubDomain::CopyGlobalToSubVector(DM gDM, DM sDM, Vec globVec, Vec subVec, const std::vector<Field>& gFields, const std::vector<Field>& subFields, bool localVector) {
    /* Get the map from the subVec to global */
    IS subpointIS;
    const PetscInt* subpointIndices = NULL;
    DMPlexGetSubpointIS(sDM, &subpointIS) >> checkError;
    ISGetIndices(subpointIS, &subpointIndices) >> checkError;

    // Get array access to the vec
    const PetscScalar* globalVecArray;
    PetscScalar* subVecArray;
    VecGetArrayRead(globVec, &globalVecArray) >> checkError;
    VecGetArray(subVec, &subVecArray) >> checkError;

    // March over the global section
    PetscSection section;
    DMGetGlobalSection(sDM, &section) >> checkError;

    PetscInt pStart, pEnd;
    PetscSectionGetChart(section, &pStart, &pEnd) >> checkError;

    // For each field in the subDM
    for (std::size_t i = 0; i < subFields.size(); i++) {
        const auto& subFieldInfo = subFields[i];
        const auto& globFieldInfo = gFields.empty() ? GetSolutionField(subFieldInfo.name) : gFields[i];

        // Get the size of the data
        PetscInt numberComponents;
        PetscSectionGetFieldComponents(section, subFieldInfo.id, &numberComponents) >> checkError;

        // March over each of the points
        for (PetscInt p = pStart; p < pEnd; p++) {
            // Get the global cell number
            PetscInt gP = subpointIndices ? subpointIndices[p] : p;

            // Hold a ref to the values
            PetscScalar* subRef;
            const PetscScalar* ref;

            if (localVector) {
                DMPlexPointLocalFieldRef(sDM, p, subFieldInfo.id, subVecArray, &subRef) >> checkError;
                DMPlexPointLocalFieldRead(gDM, gP, globFieldInfo.id, globalVecArray, &ref) >> checkError;
            } else {
                DMPlexPointGlobalFieldRef(sDM, p, subFieldInfo.id, subVecArray, &subRef) >> checkError;
                DMPlexPointGlobalFieldRead(gDM, gP, globFieldInfo.id, globalVecArray, &ref) >> checkError;
            }

            if (subRef && ref) {
                // Copy the point
                PetscArraycpy(subRef, ref, numberComponents) >> checkError;
            }
        }
    }
    VecRestoreArrayRead(globVec, &globalVecArray) >> checkError;
    VecRestoreArray(subVec, &subVecArray) >> checkError;
    ISRestoreIndices(subpointIS, &subpointIndices) >> checkError;
}
