#include "subDomain.hpp"
#include <utilities/petscError.hpp>

ablate::domain::SubDomain::SubDomain(Domain& domainIn, PetscInt dsNumber, std::vector<std::shared_ptr<FieldDescription>> allAuxFields)
    : domain(domainIn),
      name(""),
      label(nullptr),
      labelValue(0),
      fieldMap(nullptr),
      discreteSystem(nullptr),
      auxDM(nullptr),
      auxVec(nullptr),
      subDM(nullptr),
      subSolutionVec(nullptr),
      subAuxDM(nullptr),
      subAuxVec(nullptr) {
    // Get the information for this subDomain
    DMGetRegionNumDS(domain.GetDM(), dsNumber, &label, &fieldMap, &discreteSystem) >> checkError;

    // Check for the name in the label
    if (label) {
        const char* labelName;
        PetscObjectGetName((PetscObject)label, &labelName) >> checkError;
        name = std::string(labelName);
        labelValue = 1;  // assume that the region value is one for now until a different value is returned from DMGetRegionNumDS
    }

    // Get a reference to local fields
    if (fieldMap) {
        PetscInt s, e;
        const PetscInt* points;
        ISGetPointRange(fieldMap, &s, &e, &points) >> checkError;

        for (PetscInt f = s; f < e; f++) {
            PetscInt globID = points ? points[f] : f;

            const auto& field = domain.GetField(globID);
            auto newField = field.CreateSubField(f);
            fieldsByName.insert(std::make_pair(newField.name, newField));
            fieldsByType[FieldLocation::SOL].push_back(newField);
        }

    } else {
        // just copy them all over
        for (const auto& field : domain.GetFields()) {
            auto newField = field.CreateSubField(field.id);
            fieldsByName.insert(std::make_pair(newField.name, newField));
            fieldsByType[FieldLocation::SOL].push_back(newField);
        }
    }

    // Create the auxDM if there are any auxVariables in this region
    std::vector<std::shared_ptr<FieldDescription>> subAuxFields;

    // check if there is a label
    if (!label) {
        subAuxFields = allAuxFields;
    } else {
        for (auto const& auxField : allAuxFields) {
            // If there is no region, add it to the sub
            if (auxField->region == nullptr || InRegion(*auxField->region)) {
                subAuxFields.push_back(auxField);
            }
        }
    }

    // Create an aux dm if it is needed
    if (!subAuxFields.empty()) {
        /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
        DM coordDM;
        DMGetCoordinateDM(domain.GetDM(), &coordDM) >> checkError;
        DMClone(domain.GetDM(), &auxDM) >> checkError;

        // this is a hard coded "dmAux" that petsc looks for
        DMSetCoordinateDM(auxDM, coordDM) >> checkError;

        for (const auto& subAuxField : subAuxFields) {
            // Create the field and add it with the label
            auto petscField = subAuxField->CreatePetscField(domain.GetDM());

            // add to the dm
            DMAddField(auxDM, label, petscField);

            // Free the petsc after being added
            PetscObjectDestroy(&petscField);

            // Record the field
            auto newAuxField = Field::FromFieldDescription(*subAuxField, fieldsByType[FieldLocation::AUX].size(), fieldsByType[FieldLocation::AUX].size());
            fieldsByType[FieldLocation::AUX].push_back(newAuxField);
            fieldsByName.insert(std::make_pair(newAuxField.name, newAuxField));
        }
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

DM& ablate::domain::SubDomain::GetDM() { return domain.GetDM(); }

DM ablate::domain::SubDomain::GetAuxDM() { return auxDM; }

Vec ablate::domain::SubDomain::GetSolutionVector() { return domain.GetSolutionVector(); }

Vec ablate::domain::SubDomain::GetAuxVector() { return auxVec; }

PetscInt ablate::domain::SubDomain::GetDimensions() const { return domain.GetDimensions(); }

void ablate::domain::SubDomain::CreateSubDomainStructures() {
    if (auxDM) {
        DMCreateDS(auxDM) >> checkError;
        DMCreateLocalVector(auxDM, &(auxVec)) >> checkError;

        DMGetRegionDS(auxDM, label, nullptr, &auxDiscreteSystem) >> checkError;

        // attach this field as aux vector to the dm
        DMSetAuxiliaryVec(domain.GetDM(), label, labelValue, auxVec) >> checkError;
        auto vecName = "aux" + (name.empty() ? "_" + name : "");
        PetscObjectSetName((PetscObject)auxVec, vecName.c_str()) >> checkError;
    }
}

void ablate::domain::SubDomain::ProjectFieldFunctions(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& initialization, Vec globVec, PetscReal time) {
    PetscInt numberFields;
    DM dm;

    VecGetDM(globVec, &dm) >> checkError;
    if (dm != GetDM()) {
        throw std::invalid_argument("The Vector passed in to ablate::domain::SubDomain::ProjectFieldFunctions must be a global vector from the DM.");
    }
    DMGetNumFields(dm, &numberFields) >> checkError;

    // size up the update and context functions
    std::vector<mathFunctions::PetscFunction> fieldFunctions(numberFields, nullptr);
    std::vector<void*> fieldContexts(numberFields, nullptr);

    for (auto& fieldInitialization : initialization) {
        auto fieldId = GetField(fieldInitialization->GetName());

        fieldContexts[fieldId.id] = fieldInitialization->GetSolutionField().GetContext();
        fieldFunctions[fieldId.id] = fieldInitialization->GetSolutionField().GetPetscFunction();
    }

    if (label == nullptr) {
        DMProjectFunction(dm, time, &fieldFunctions[0], &fieldContexts[0], INSERT_VALUES, globVec) >> checkError;
    } else {
        DMProjectFunctionLabel(dm, time, label, 1, &labelValue, -1, nullptr, &fieldFunctions[0], &fieldContexts[0], INSERT_VALUES, globVec);
    }
}

void ablate::domain::SubDomain::ProjectFieldFunctionsToSubDM(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& initialization, Vec globVec, PetscReal time) {
    if (subDM == nullptr) {
        return ProjectFieldFunctions(initialization, globVec, time);
    }

    PetscInt numberFields;
    DM dm;

    VecGetDM(globVec, &dm) >> checkError;
    if (dm != subDM) {
        throw std::invalid_argument("The Vector passed in to ablate::domain::SubDomain::ProjectFieldFunctionsToSubDM must be a global vector from the SubDM.");
    }
    DMGetNumFields(subDM, &numberFields) >> checkError;

    // size up the update and context functions
    std::vector<mathFunctions::PetscFunction> fieldFunctions(numberFields, nullptr);
    std::vector<void*> fieldContexts(numberFields, nullptr);

    for (auto& fieldInitialization : initialization) {
        auto fieldId = GetField(fieldInitialization->GetName());

        fieldContexts[fieldId.subId] = fieldInitialization->GetSolutionField().GetContext();
        fieldFunctions[fieldId.subId] = fieldInitialization->GetSolutionField().GetPetscFunction();
    }

    DMProjectFunction(subDM, time, &fieldFunctions[0], &fieldContexts[0], INSERT_VALUES, globVec) >> checkError;
}

DM ablate::domain::SubDomain::GetSubDM() {
    // If there is no label, just return the entire dm
    if (!label) {
        return GetDM();
    }

    // If the subDM has not been created, create one
    if (!subDM) {
        // filter by label
        DMPlexFilter(GetDM(), label, labelValue, &subDM) >> checkError;

        // copy over all fields that were in the main dm
        for (auto& fieldInfo : GetFields()) {
            auto petscField = GetPetscFieldObject(fieldInfo);
            DMAddField(subDM, nullptr, petscField) >> checkError;
        }

        DMCreateDS(subDM) >> checkError;
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
    for (auto& fieldInfo : GetFields(FieldLocation::AUX)) {
        auto petscField = GetPetscFieldObject(fieldInfo);
        DMAddField(subAuxDM, nullptr, petscField) >> checkError;
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

        // Copy the aux vector name
        const char* vecName;
        PetscObjectGetName((PetscObject)GetAuxVector(), &vecName) >> checkError;
        PetscObjectSetName((PetscObject)subAuxVec, vecName) >> checkError;
    }

    CopyGlobalToSubVector(GetSubAuxDM(), GetAuxDM(), subAuxVec, GetAuxVector(), GetFields(FieldLocation::AUX), GetFields(FieldLocation::AUX), true);
    return subAuxVec;
}

void ablate::domain::SubDomain::CopyGlobalToSubVector(DM sDM, DM gDM, Vec subVec, Vec globVec, const std::vector<Field>& subFields, const std::vector<Field>& gFields, bool localVector) const {
    /* Get the map from the subVec to global */
    IS subpointIS;
    const PetscInt* subpointIndices = nullptr;
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
        const auto& globFieldInfo = gFields.empty() ? GetField(subFieldInfo.name) : gFields[i];

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

bool ablate::domain::SubDomain::InRegion(const domain::Region& region) const {
    if (!label) {
        return true;
    }
    bool inside = false;
    // Check to see if this region is inside of the dsLabel
    DMLabel regionLabel;
    DMGetLabel(domain.GetDM(), region.GetName().c_str(), &regionLabel) >> checkError;
    if (!regionLabel) {
        throw std::invalid_argument("Label " + region.GetName() + " not found ");
    }

    // Get the is from the dsLabel
    IS dsLabelIS;
    DMLabelGetStratumIS(label, 1, &dsLabelIS) >> checkError;

    // Get the IS for this label
    IS regionIS;
    DMLabelGetStratumIS(regionLabel, region.GetValue(), &regionIS) >> checkError;

    // Check for an overlap
    IS overlapIS;
    ISIntersect(dsLabelIS, regionIS, &overlapIS) >> checkError;

    // determine if there is an overlap
    if (overlapIS) {
        PetscInt size;
        ISGetSize(overlapIS, &size) >> checkError;
        if (size) {
            inside = true;
        }
        ISDestroy(&overlapIS) >> checkError;
    }
    ISDestroy(&regionIS) >> checkError;
    ISDestroy(&dsLabelIS) >> checkError;
    return inside;
}
PetscObject ablate::domain::SubDomain::GetPetscFieldObject(const ablate::domain::Field& field) {
    switch (field.location) {
        case FieldLocation::SOL: {
            PetscObject fieldObject;
            DMGetField(GetDM(), field.id, nullptr, &fieldObject) >> checkError;
            return fieldObject;
        }
        case FieldLocation::AUX: {
            PetscObject fieldObject;
            DMGetField(auxDM, field.subId, nullptr, &fieldObject) >> checkError;
            return fieldObject;
        }
        default:
            throw std::invalid_argument("Unknown field location for " + field.name);
    }
}
