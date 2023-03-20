#include "subDomain.hpp"
#include <set>
#include <sstream>
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscUtilities.hpp"

ablate::domain::SubDomain::SubDomain(Domain& domainIn, PetscInt dsNumber, const std::vector<std::shared_ptr<FieldDescription>>& allAuxFields)
    : domain(domainIn),
      name(defaultName),
      label(nullptr),
      labelValue(0),
      fieldMap(nullptr),
      discreteSystem(nullptr),
      auxDM(nullptr),
      auxLocalVec(nullptr),
      auxGlobalVec(nullptr),
      subDM(nullptr),
      subSolutionVec(nullptr),
      subAuxDM(nullptr),
      subAuxVec(nullptr) {
    // Get the information for this subDomain
    DMGetRegionNumDS(domain.GetDM(), dsNumber, &label, &fieldMap, &discreteSystem) >> utilities::PetscUtilities::checkError;

    // Check for the name in the label
    if (label) {
        const char* labelName;
        PetscObjectGetName((PetscObject)label, &labelName) >> utilities::PetscUtilities::checkError;
        name = std::string(labelName);
        labelValue = 1;  // assume that the region value is one for now until a different value is returned from DMGetRegionNumDS
    }

    // Get a reference to local fields
    if (fieldMap) {
        PetscInt s, e;
        const PetscInt* points;
        ISGetPointRange(fieldMap, &s, &e, &points) >> utilities::PetscUtilities::checkError;

        for (PetscInt f = s; f < e; f++) {
            PetscInt globID = points ? points[f] : f;

            // Get the offset for this field
            PetscInt fieldOffset;
            PetscDSGetFieldOffset(discreteSystem, f, &fieldOffset) >> utilities::PetscUtilities::checkError;

            const auto& field = domain.GetField(globID);
            auto newField = field.CreateSubField(f, fieldOffset);
            fieldsByName.insert(std::make_pair(newField.name, newField));
            fieldsByType[FieldLocation::SOL].push_back(newField);
        }

    } else {
        // just copy them all over
        for (const auto& field : domain.GetFields()) {
            // Get the offset for this field
            PetscInt fieldOffset;
            PetscDSGetFieldOffset(discreteSystem, field.id, &fieldOffset) >> utilities::PetscUtilities::checkError;

            auto newField = field.CreateSubField(field.id, fieldOffset);
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
        DMGetCoordinateDM(domain.GetDM(), &coordDM) >> utilities::PetscUtilities::checkError;
        DMClone(domain.GetDM(), &auxDM) >> utilities::PetscUtilities::checkError;

        // this is a hard coded "dmAux" that petsc looks for
        DMSetCoordinateDM(auxDM, coordDM) >> utilities::PetscUtilities::checkError;

        // Keep track of the offset so it can be computed upfront
        PetscInt offset = 0;

        for (const auto& subAuxField : subAuxFields) {
            // Create the field and add it with the label
            auto petscField = subAuxField->CreatePetscField(domain.GetDM());

            // add to the dm
            DMAddField(auxDM, label, petscField);

            // Free the petsc after being added
            PetscObjectDestroy(&petscField);

            // Record the field
            auto newAuxField = Field::FromFieldDescription(*subAuxField, (PetscInt)fieldsByType[FieldLocation::AUX].size(), (PetscInt)fieldsByType[FieldLocation::AUX].size(), offset);
            fieldsByType[FieldLocation::AUX].push_back(newAuxField);
            fieldsByName.insert(std::make_pair(newAuxField.name, newAuxField));
            offset += newAuxField.numberComponents;
        }
    }
}

ablate::domain::SubDomain::~SubDomain() {
    if (auxDM) {
        DMDestroy(&auxDM) >> utilities::PetscUtilities::checkError;
    }

    if (auxLocalVec) {
        VecDestroy(&auxLocalVec) >> utilities::PetscUtilities::checkError;
    }

    if (auxGlobalVec) {
        VecDestroy(&auxGlobalVec) >> utilities::PetscUtilities::checkError;
    }

    if (subDM) {
        DMDestroy(&subDM) >> utilities::PetscUtilities::checkError;
    }

    if (subSolutionVec) {
        VecDestroy(&subSolutionVec) >> utilities::PetscUtilities::checkError;
    }

    if (subAuxDM) {
        DMDestroy(&subAuxDM) >> utilities::PetscUtilities::checkError;
    }

    if (subAuxVec) {
        VecDestroy(&subAuxVec) >> utilities::PetscUtilities::checkError;
    }
}

void ablate::domain::SubDomain::CreateSubDomainStructures() {
    if (auxDM) {
        DMCreateDS(auxDM) >> utilities::PetscUtilities::checkError;
        DMCreateLocalVector(auxDM, &(auxLocalVec)) >> utilities::PetscUtilities::checkError;
        DMCreateGlobalVector(auxDM, &(auxGlobalVec)) >> utilities::PetscUtilities::checkError;

        DMGetRegionDS(auxDM, label, nullptr, &auxDiscreteSystem) >> utilities::PetscUtilities::checkError;

        // attach this field as aux vector to the dm
        DMSetAuxiliaryVec(domain.GetDM(), label, labelValue, 0 /*The equation part, or 0 if unused*/, auxLocalVec) >> utilities::PetscUtilities::checkError;
        PetscObjectSetName((PetscObject)auxLocalVec, "aux") >> utilities::PetscUtilities::checkError;
        PetscObjectSetName((PetscObject)auxGlobalVec, "aux") >> utilities::PetscUtilities::checkError;

        // add the names to each of the components in the dm section
        PetscSection section;
        DMGetLocalSection(auxDM, &section) >> utilities::PetscUtilities::checkError;
        for (const auto& field : GetFields(FieldLocation::AUX)) {
            if (field.numberComponents > 1) {
                for (PetscInt c = 0; c < field.numberComponents; c++) {
                    PetscSectionSetComponentName(section, field.id, c, field.components[c].c_str()) >> utilities::PetscUtilities::checkError;
                }
            }
        }
    }
}

void ablate::domain::SubDomain::ProjectFieldFunctionsToSubDM(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& initialization, Vec globVec, PetscReal time) {
    if (subDM == nullptr) {
        return domain.ProjectFieldFunctions(initialization, globVec, time);
    }

    PetscInt numberFields;
    DM dm;

    VecGetDM(globVec, &dm) >> utilities::PetscUtilities::checkError;
    if (dm != subDM) {
        throw std::invalid_argument("The Vector passed in to ablate::domain::SubDomain::ProjectFieldFunctionsToSubDM must be a global vector from the SubDM.");
    }
    DMGetNumFields(subDM, &numberFields) >> utilities::PetscUtilities::checkError;

    // get a local vector for the work
    Vec locVec;
    DMGetLocalVector(dm, &locVec) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocal(dm, globVec, INSERT_VALUES, locVec) >> utilities::PetscUtilities::checkError;

    // size up the update and context functions
    std::vector<mathFunctions::PetscFunction> fieldFunctions(numberFields, nullptr);
    std::vector<void*> fieldContexts(numberFields, nullptr);

    for (auto& fieldInitialization : initialization) {
        auto fieldId = GetField(fieldInitialization->GetName());

        fieldContexts[fieldId.subId] = fieldInitialization->GetSolutionField().GetContext();
        fieldFunctions[fieldId.subId] = fieldInitialization->GetSolutionField().GetPetscFunction();
    }

    DMProjectFunctionLocal(subDM, time, &fieldFunctions[0], &fieldContexts[0], INSERT_VALUES, locVec) >> utilities::PetscUtilities::checkError;

    // push the results back to the global vector
    DMLocalToGlobal(dm, locVec, INSERT_VALUES, globVec) >> utilities::PetscUtilities::checkError;
    DMRestoreLocalVector(dm, &locVec) >> utilities::PetscUtilities::checkError;
}

Vec ablate::domain::SubDomain::GetAuxGlobalVector() {
    if (!auxDM) {
        return nullptr;
    }
    DMLocalToGlobal(auxDM, auxLocalVec, INSERT_VALUES, auxGlobalVec) >> utilities::PetscUtilities::checkError;
    return auxGlobalVec;
}

DM ablate::domain::SubDomain::GetSubDM() {
    // If there is no label, just return the entire dm
    if (!label) {
        return GetDM();
    }

    // If the subDM has not been created, create one
    if (!subDM) {
        // filter by label
        DMPlexFilter(GetDM(), label, labelValue, &subDM) >> utilities::PetscUtilities::checkError;

        // copy over all fields that were in the main dm
        for (auto& fieldInfo : GetFields()) {
            auto petscField = GetPetscFieldObject(fieldInfo);
            DMAddField(subDM, nullptr, petscField) >> utilities::PetscUtilities::checkError;
        }

        // add the names to each of the components in the dm section
        PetscSection section;
        DMGetLocalSection(subDM, &section) >> utilities::PetscUtilities::checkError;
        for (const auto& field : GetFields()) {
            if (field.numberComponents > 1) {
                for (PetscInt c = 0; c < field.numberComponents; c++) {
                    PetscSectionSetComponentName(section, field.id, c, field.components[c].c_str()) >> utilities::PetscUtilities::checkError;
                }
            }
        }

        DMCreateDS(subDM) >> utilities::PetscUtilities::checkError;

        // Copy over options
        PetscOptions options;
        PetscObjectGetOptions((PetscObject)GetDM(), &options) >> utilities::PetscUtilities::checkError;
        PetscObjectSetOptions((PetscObject)subDM, options) >> utilities::PetscUtilities::checkError;
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
        DMCreateGlobalVector(subDM, &subSolutionVec) >> utilities::PetscUtilities::checkError;

        // Assign a name the same as the global
        const char* vecName;
        PetscObjectGetName((PetscObject)GetSolutionVector(), &vecName) >> utilities::PetscUtilities::checkError;
        PetscObjectSetName((PetscObject)subSolutionVec, vecName) >> utilities::PetscUtilities::checkError;
    }

    // Make a local version of the vector
    Vec localSolutionVec;
    DMGetLocalVector(GetDM(), &localSolutionVec) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocalBegin(GetDM(), GetSolutionVector(), INSERT_VALUES, localSolutionVec) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocalEnd(GetDM(), GetSolutionVector(), INSERT_VALUES, localSolutionVec) >> utilities::PetscUtilities::checkError;

    CopyGlobalToSubVector(subDM, GetDM(), subSolutionVec, localSolutionVec, GetFields(), GetFields(), true);
    DMRestoreLocalVector(GetDM(), &localSolutionVec) >> utilities::PetscUtilities::checkError;
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
    DMGetCoordinateDM(GetSubDM(), &coordDM) >> utilities::PetscUtilities::checkError;
    DMClone(GetSubDM(), &subAuxDM) >> utilities::PetscUtilities::checkError;

    // this is a hard coded "dmAux" that petsc looks for
    DMSetCoordinateDM(subAuxDM, coordDM) >> utilities::PetscUtilities::checkError;

    // Add all of the fields
    // copy over all fields that were in the main dm
    for (auto& fieldInfo : GetFields(FieldLocation::AUX)) {
        auto petscField = GetPetscFieldObject(fieldInfo);
        DMAddField(subAuxDM, nullptr, petscField) >> utilities::PetscUtilities::checkError;
    }

    // add the names to each of the components in the dm section
    PetscSection section;
    DMGetLocalSection(subAuxDM, &section) >> utilities::PetscUtilities::checkError;
    for (const auto& field : GetFields(FieldLocation::AUX)) {
        if (field.numberComponents > 1) {
            for (PetscInt c = 0; c < field.numberComponents; c++) {
                PetscSectionSetComponentName(section, field.id, c, field.components[c].c_str()) >> utilities::PetscUtilities::checkError;
            }
        }
    }

    return subAuxDM;
}

Vec ablate::domain::SubDomain::GetSubAuxVector() {
    // If there is no auxVector, return null
    if (!auxLocalVec) {
        return nullptr;
    }

    // If there is no label, just return the entire solution vector
    if (!label) {
        // The subAuxVector is always treated as a global vector
        return GetAuxGlobalVector();
    }

    if (!subAuxVec) {
        DMCreateGlobalVector(GetSubAuxDM(), &subAuxVec) >> utilities::PetscUtilities::checkError;

        // Copy the aux vector name
        const char* vecName;
        PetscObjectGetName((PetscObject)GetAuxVector(), &vecName) >> utilities::PetscUtilities::checkError;
        PetscObjectSetName((PetscObject)subAuxVec, vecName) >> utilities::PetscUtilities::checkError;
    }

    CopyGlobalToSubVector(GetSubAuxDM(), GetAuxDM(), subAuxVec, GetAuxVector(), GetFields(FieldLocation::AUX), GetFields(FieldLocation::AUX), true);
    return subAuxVec;
}

void ablate::domain::SubDomain::CopyGlobalToSubVector(DM sDM, DM gDM, Vec subVec, Vec globVec, const std::vector<Field>& subFields, const std::vector<Field>& gFields, bool localVector) const {
    /* Get the map from the subVec to global */
    IS subpointIS;
    const PetscInt* subpointIndices = nullptr;
    DMPlexGetSubpointIS(sDM, &subpointIS) >> utilities::PetscUtilities::checkError;
    ISGetIndices(subpointIS, &subpointIndices) >> utilities::PetscUtilities::checkError;

    // Get array access to the vec
    const PetscScalar* globalVecArray;
    PetscScalar* subVecArray;
    VecGetArrayRead(globVec, &globalVecArray) >> utilities::PetscUtilities::checkError;
    VecGetArray(subVec, &subVecArray) >> utilities::PetscUtilities::checkError;

    // March over the global section
    PetscSection section;
    DMGetLocalSection(sDM, &section) >> utilities::PetscUtilities::checkError;

    PetscInt pStart, pEnd;
    PetscSectionGetChart(section, &pStart, &pEnd) >> utilities::PetscUtilities::checkError;

    // For each field in the subDM
    for (std::size_t i = 0; i < subFields.size(); i++) {
        const auto& subFieldInfo = subFields[i];
        const auto& globFieldInfo = gFields.empty() ? GetField(subFieldInfo.name) : gFields[i];

        // Get the size of the data
        PetscInt numberComponents;
        PetscSectionGetFieldComponents(section, subFieldInfo.id, &numberComponents) >> utilities::PetscUtilities::checkError;

        // March over each of the points
        for (PetscInt p = pStart; p < pEnd; p++) {
            // Get the global cell number
            PetscInt gP = subpointIndices ? subpointIndices[p] : p;

            // Hold a ref to the values
            PetscScalar* subRef = nullptr;
            const PetscScalar* ref = nullptr;

            DMPlexPointGlobalFieldRef(sDM, p, subFieldInfo.id, subVecArray, &subRef) >> utilities::PetscUtilities::checkError;
            if (localVector) {
                DMPlexPointLocalFieldRead(gDM, gP, globFieldInfo.id, globalVecArray, &ref) >> utilities::PetscUtilities::checkError;
            } else {
                DMPlexPointGlobalFieldRead(gDM, gP, globFieldInfo.id, globalVecArray, &ref) >> utilities::PetscUtilities::checkError;
            }

            if (subRef && ref) {
                // Copy the point
                PetscArraycpy(subRef, ref, numberComponents) >> utilities::PetscUtilities::checkError;
            }
        }
    }
    VecRestoreArrayRead(globVec, &globalVecArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArray(subVec, &subVecArray) >> utilities::PetscUtilities::checkError;
    ISRestoreIndices(subpointIS, &subpointIndices) >> utilities::PetscUtilities::checkError;
}

void ablate::domain::SubDomain::CopySubVectorToGlobal(DM sDM, DM gDM, Vec subVec, Vec globVec, const std::vector<Field>& subFields, const std::vector<Field>& gFields, bool localVector) const {
    /* Get the map from the subVec to global */
    IS subpointIS;
    const PetscInt* subpointIndices = nullptr;
    DMPlexGetSubpointIS(sDM, &subpointIS) >> utilities::PetscUtilities::checkError;
    ISGetIndices(subpointIS, &subpointIndices) >> utilities::PetscUtilities::checkError;

    // Get array access to the vec
    PetscScalar* globalVecArray;
    const PetscScalar* subVecArray;
    VecGetArray(globVec, &globalVecArray) >> utilities::PetscUtilities::checkError;
    VecGetArrayRead(subVec, &subVecArray) >> utilities::PetscUtilities::checkError;

    // March over the global section
    PetscSection section;
    DMGetGlobalSection(sDM, &section) >> utilities::PetscUtilities::checkError;

    PetscInt pStart, pEnd;
    PetscSectionGetChart(section, &pStart, &pEnd) >> utilities::PetscUtilities::checkError;

    // For each field in the subDM
    for (std::size_t i = 0; i < subFields.size(); i++) {
        const auto& subFieldInfo = subFields[i];
        const auto& globFieldInfo = gFields.empty() ? GetField(subFieldInfo.name) : gFields[i];

        // Get the size of the data
        PetscInt numberComponents;
        PetscSectionGetFieldComponents(section, subFieldInfo.id, &numberComponents) >> utilities::PetscUtilities::checkError;

        // March over each of the points
        for (PetscInt p = pStart; p < pEnd; p++) {
            // Get the global cell number
            PetscInt gP = subpointIndices ? subpointIndices[p] : p;

            // Hold a ref to the values
            const PetscScalar* subRef = nullptr;
            PetscScalar* ref = nullptr;

            DMPlexPointGlobalFieldRead(sDM, p, subFieldInfo.id, subVecArray, &subRef) >> utilities::PetscUtilities::checkError;
            if (localVector) {
                DMPlexPointLocalFieldRef(gDM, gP, globFieldInfo.id, globalVecArray, &ref) >> utilities::PetscUtilities::checkError;
            } else {
                DMPlexPointGlobalFieldRef(gDM, gP, globFieldInfo.id, globalVecArray, &ref) >> utilities::PetscUtilities::checkError;
            }

            if (subRef && ref) {
                // Copy the point
                PetscArraycpy(ref, subRef, numberComponents) >> utilities::PetscUtilities::checkError;
            }
        }
    }
    VecRestoreArray(globVec, &globalVecArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(subVec, &subVecArray) >> utilities::PetscUtilities::checkError;
    ISRestoreIndices(subpointIS, &subpointIndices) >> utilities::PetscUtilities::checkError;
}

bool ablate::domain::SubDomain::InRegion(const domain::Region& region) const {
    if (!label) {
        return true;
    }

    // Compute the size of region inside the subDomain label
    PetscInt size = 0;

    // Check to see if this region is inside of the dsLabel
    DMLabel regionLabel;
    DMGetLabel(domain.GetDM(), region.GetName().c_str(), &regionLabel) >> utilities::PetscUtilities::checkError;
    if (!regionLabel) {
        throw std::invalid_argument("Label " + region.GetName() + " not found ");
    }

    // Get the is from the dsLabel
    IS dsLabelIS;
    DMLabelGetStratumIS(label, 1, &dsLabelIS) >> utilities::PetscUtilities::checkError;

    // Get the IS for this label
    IS regionIS;
    DMLabelGetStratumIS(regionLabel, region.GetValue(), &regionIS) >> utilities::PetscUtilities::checkError;

    // each rank must check separately and then share the result
    if (regionIS) {
        // Check for an overlap
        IS overlapIS;
        ISIntersect(dsLabelIS, regionIS, &overlapIS) >> utilities::PetscUtilities::checkError;

        // determine if there is an overlap
        if (overlapIS) {
            ISGetSize(overlapIS, &size) >> utilities::PetscUtilities::checkError;
            ISDestroy(&overlapIS) >> utilities::PetscUtilities::checkError;
        }
        ISDestroy(&regionIS) >> utilities::PetscUtilities::checkError;
    }
    ISDestroy(&dsLabelIS) >> utilities::PetscUtilities::checkError;

    // Take the sum
    PetscInt globalSize;
    MPI_Allreduce(&size, &globalSize, 1, MPIU_INT, MPIU_SUM, GetComm()) >> ablate::utilities::MpiUtilities::checkError;
    return globalSize > 0;
}

PetscObject ablate::domain::SubDomain::GetPetscFieldObject(const ablate::domain::Field& field) {
    switch (field.location) {
        case FieldLocation::SOL: {
            PetscObject fieldObject;
            DMGetField(GetDM(), field.id, nullptr, &fieldObject) >> utilities::PetscUtilities::checkError;
            return fieldObject;
        }
        case FieldLocation::AUX: {
            PetscObject fieldObject;
            DMGetField(auxDM, field.subId, nullptr, &fieldObject) >> utilities::PetscUtilities::checkError;
            return fieldObject;
        }
        default:
            throw std::invalid_argument("Unknown field location for " + field.name);
    }
}

PetscErrorCode ablate::domain::SubDomain::GetFieldGlobalVector(const Field& field, IS* vecIs, Vec* vec, DM* subdm) {
    PetscFunctionBeginUser;
    // Get the correct dm
    auto entireDm = GetFieldDM(field);
    auto entireVec = GetGlobalVec(field);

    PetscCall(DMCreateSubDM(entireDm, 1, &field.id, vecIs, subdm));

    // Get the sub vector
    PetscCall(VecGetSubVector(entireVec, *vecIs, vec));
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::domain::SubDomain::RestoreFieldGlobalVector(const Field& field, IS* vecIs, Vec* vec, DM* subdm) {
    PetscFunctionBeginUser;
    auto entireVec = GetGlobalVec(field);

    PetscCall(VecRestoreSubVector(entireVec, *vecIs, vec));
    PetscCall(ISDestroy(vecIs));
    PetscCall(DMDestroy(subdm));

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::domain::SubDomain::GetFieldLocalVector(const ablate::domain::Field& field, PetscReal time, IS* vecIs, Vec* vec, DM* subdm) {
    PetscFunctionBeginUser;

    if (field.location == FieldLocation::SOL) {
        // Get the correct dm
        auto entireDm = GetDM();
        auto entireVec = GetSolutionVector();

        // Create a subDM
        PetscCall(DMCreateSubDM(entireDm, 1, &field.id, vecIs, subdm));

        // Use a global vector to get the results
        Vec subGlobalVector;
        PetscCall(VecGetSubVector(entireVec, *vecIs, &subGlobalVector));

        // Make a local version of the vector
        PetscCall(DMGetLocalVector(*subdm, vec));
        PetscCall(DMPlexInsertBoundaryValues(*subdm, PETSC_TRUE, *vec, time, nullptr, nullptr, nullptr));
        PetscCall(DMGlobalToLocalBegin(*subdm, subGlobalVector, INSERT_VALUES, *vec));
        PetscCall(DMGlobalToLocalEnd(*subdm, subGlobalVector, INSERT_VALUES, *vec));

        // We have the filled local vec subdm, so clean up the subGlobalVector and vecIS
        PetscCall(VecRestoreSubVector(entireVec, *vecIs, &subGlobalVector));
        PetscCall(ISDestroy(vecIs); *vecIs = nullptr);
    } else if (field.location == FieldLocation::AUX) {
        auto entireDm = GetAuxDM();
        auto entireVec = GetAuxVector();

        PetscCall(DMCreateSubDM(entireDm, 1, &field.id, vecIs, subdm));

        // Get the sub vector
        PetscCall(VecGetSubVector(entireVec, *vecIs, vec));
    } else {
        SETERRQ(GetComm(), PETSC_ERR_SUP, "%s", "Unknown field location");
    }

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::domain::SubDomain::RestoreFieldLocalVector(const ablate::domain::Field& field, IS* vecIs, Vec* vec, DM* subdm) {
    PetscFunctionBeginUser;

    if (field.location == FieldLocation::SOL) {
        // In the Get call, the vecIS was already cleaned up and vec is only a localVec
        PetscCall(DMRestoreLocalVector(*subdm, vec));
        PetscCall(DMDestroy(subdm));
    } else if (field.location == FieldLocation::AUX) {
        auto entireVec = GetAuxVector();
        PetscCall(VecRestoreSubVector(entireVec, *vecIs, vec));
        PetscCall(ISDestroy(vecIs));
        PetscCall(DMDestroy(subdm));
    } else {
        SETERRQ(GetComm(), PETSC_ERR_SUP, "%s", "Unknown field location");
    }

    PetscFunctionReturn(0);
}

void ablate::domain::SubDomain::SetsExactSolutions(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& exactSolutionsIn) {
    // if an exact solution has been provided register it
    for (const auto& exactSolution : exactSolutionsIn) {
        // check to see if this field in is in this subDomain
        if (ContainsField(exactSolution->GetName())) {
            // store it
            exactSolutions.push_back(exactSolution);

            // Get the field information
            auto fieldInfo = GetField(exactSolution->GetName());

            if (exactSolution->HasSolutionField()) {
                PetscDSSetExactSolution(GetDiscreteSystem(), fieldInfo.subId, exactSolution->GetSolutionField().GetPetscFunction(), exactSolution->GetSolutionField().GetContext()) >>
                    utilities::PetscUtilities::checkError;
            }
            if (exactSolution->HasTimeDerivative()) {
                PetscDSSetExactSolutionTimeDerivative(GetDiscreteSystem(), fieldInfo.subId, exactSolution->GetTimeDerivative().GetPetscFunction(), exactSolution->GetTimeDerivative().GetContext()) >>
                    utilities::PetscUtilities::checkError;
            }
        }
    }
}
void ablate::domain::SubDomain::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    PetscFunctionBeginUser;
    auto locSubDm = GetSubDM();
    auto locAuxDM = GetSubAuxDM();
    // If this is the first output, save the mesh
    if (sequenceNumber == 0) {
        // Print the initial mesh
        DMView(locSubDm, viewer) >> utilities::PetscUtilities::checkError;
    }

    // set the dm sequence number, because we may be skipping outputs
    DMSetOutputSequenceNumber(locSubDm, sequenceNumber, time) >> utilities::PetscUtilities::checkError;
    if (locAuxDM) {
        DMSetOutputSequenceNumber(locAuxDM, sequenceNumber, time) >> utilities::PetscUtilities::checkError;
    }

    // Always save the main flowField
    VecView(GetSubSolutionVector(), viewer) >> utilities::PetscUtilities::checkError;

    // If there is aux data output
    PetscBool outputAuxVector = PETSC_TRUE;
    PetscOptionsGetBool(nullptr, nullptr, "-outputAuxVector", &outputAuxVector, nullptr) >> utilities::PetscUtilities::checkError;
    if (outputAuxVector) {
        if (auto subAuxVector = GetSubAuxVector()) {
            // copy over the sequence data from the main dm
            PetscReal dmTime;
            PetscInt dmSequence;
            DMGetOutputSequenceNumber(locSubDm, &dmSequence, &dmTime) >> utilities::PetscUtilities::checkError;
            DMSetOutputSequenceNumber(locAuxDM, dmSequence, dmTime) >> utilities::PetscUtilities::checkError;

            VecView(subAuxVector, viewer) >> utilities::PetscUtilities::checkError;
        }
    }

    // If there is an exact solution save it
    if (!exactSolutions.empty()) {
        Vec exactVec;
        DMGetGlobalVector(GetSubDM(), &exactVec) >> utilities::PetscUtilities::checkError;

        ProjectFieldFunctionsToSubDM(exactSolutions, exactVec, time);

        PetscObjectSetName((PetscObject)exactVec, "exact") >> utilities::PetscUtilities::checkError;
        VecView(exactVec, viewer) >> utilities::PetscUtilities::checkError;
        DMRestoreGlobalVector(GetSubDM(), &exactVec) >> utilities::PetscUtilities::checkError;
    }
    PetscFunctionReturnVoid();
}
void ablate::domain::SubDomain::Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    // The only item that needs to be explicitly restored is the flowField
    DMSetOutputSequenceNumber(GetDM(), sequenceNumber, time) >> utilities::PetscUtilities::checkError;
    DMSetOutputSequenceNumber(GetSubDM(), sequenceNumber, time) >> utilities::PetscUtilities::checkError;
    auto solutionVector = GetSubSolutionVector();
    VecLoad(solutionVector, viewer) >> utilities::PetscUtilities::checkError;

    // copy back if needed
    if (subSolutionVec == solutionVector) {
        CopySubVectorToGlobal(subDM, GetDM(), subSolutionVec, GetSolutionVector(), GetFields());
    }
}
void ablate::domain::SubDomain::ProjectFieldFunctionsToLocalVector(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& fieldFunctions, Vec locVec, PetscReal time) const {
    PetscInt numberFields;
    DM dm;

    VecGetDM(locVec, &dm) >> utilities::PetscUtilities::checkError;
    DMGetNumFields(dm, &numberFields) >> utilities::PetscUtilities::checkError;

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
            DMGetLabel(dm, region->GetName().c_str(), &fieldLabel) >> utilities::PetscUtilities::checkError;
        } else {
            PetscObject fieldTemp;
            DMGetField(dm, fieldId.id, &fieldLabel, &fieldTemp) >> utilities::PetscUtilities::checkError;
            if (fieldLabel) {
                fieldValue = 1;  // this is temporary until petsc allows fields to be defined with values beside 1
            }
        }

        // Note the global DMProjectFunctionLabel can't be used because it overwrites unwritten values.
        // Project this field
        if (fieldLabel) {
            DMProjectFunctionLabelLocal(dm, time, fieldLabel, 1, &fieldValue, -1, nullptr, fieldFunctionsPts.data(), fieldContexts.data(), INSERT_VALUES, locVec) >>
                utilities::PetscUtilities::checkError;
        } else {
            DMProjectFunctionLocal(dm, time, fieldFunctionsPts.data(), fieldContexts.data(), INSERT_VALUES, locVec) >> utilities::PetscUtilities::checkError;
        }
    }
}

void ablate::domain::SubDomain::CreateEmptySubDM(DM* inDM, std::shared_ptr<domain::Region> region) {
    DMLabel subDmLabel = nullptr;
    PetscInt subDmValue;
    if (region) {
        // Get the region info from the provided region
        domain::Region::GetLabel(region, GetDM(), subDmLabel, subDmValue);
    } else {
        // Grab it from the domain itself
        subDmLabel = GetLabel();
        subDmValue = labelValue;
    }
    if (subDmLabel) {
        DMPlexFilter(GetDM(), subDmLabel, subDmValue, inDM);
    } else {
        DMClone(GetDM(), inDM);
    }
}

std::vector<ablate::domain::Field> ablate::domain::SubDomain::GetFields(ablate::domain::FieldLocation type, std::string_view tag) const {
    std::vector<ablate::domain::Field> taggedFields;
    const auto& fields = fieldsByType.at(type);
    std::copy_if(fields.begin(), fields.end(), std::back_inserter(taggedFields), [tag](const auto& field) {
        return std::any_of(field.tags.begin(), field.tags.end(), [tag](const auto& tagItem) { return tagItem == tag; });
    });

    return taggedFields;
}


void ablate::domain::SubDomain::GetRange(const std::shared_ptr<ablate::domain::Region> region, PetscInt depth, ablate::solver::Range &range) {
    // Start out getting all the points
    IS allPointIS;
    DMGetStratumIS(subDomain->GetDM(), "dim", depth, &allPointIS) >> utilities::PetscUtilities::checkError;
    if (!allPointIS) {
        DMGetStratumIS(subDomain->GetDM(), "depth", depth, &allPointIS) >> utilities::PetscUtilities::checkError;
    }

    // If there is a label for this solver, get only the parts of the mesh that here
    if (region) {
        DMLabel label;
        DMGetLabel(subDomain->GetDM(), region->GetName().c_str(), &label);

        IS labelIS;
        DMLabelGetStratumIS(label, region->GetValue(), &labelIS) >> utilities::PetscUtilities::checkError;
        ISIntersect_Caching_Internal(allPointIS, labelIS, &range.is) >> utilities::PetscUtilities::checkError;
        ISDestroy(&labelIS) >> utilities::PetscUtilities::checkError;
    } else {
        PetscObjectReference((PetscObject)allPointIS) >> utilities::PetscUtilities::checkError;
        range.is = allPointIS;
    }

    // Get the point range
    if (range.is == nullptr) {
        // There are no points in this region, so skip
        range.start = 0;
        range.end = 0;
        range.points = nullptr;
    } else {
        // Get the range
        ISGetPointRange(range.is, &range.start, &range.end, &range.points) >> utilities::PetscUtilities::checkError;
    }

    // Clean up the allCellIS
    ISDestroy(&allPointIS) >> utilities::PetscUtilities::checkError;
}

void ablate::domain::SubDomain::GetCellRange(const std::shared_ptr<ablate::domain::Region> region, ablate::solver::Range &cellRange) {
    // Start out getting all the cells
    PetscInt depth;
    DMPlexGetDepth(subDomain->GetDM(), &depth) >> utilities::PetscUtilities::checkError;
    RBF::GetRange(region, depth, cellRange);
}

void ablate::domain::SubDomain::RestoreRange(ablate::solver::Range &range) {
    if (range.is) {
        ISRestorePointRange(range.is, &range.start, &range.end, &range.points) >> utilities::PetscUtilities::checkError;
        ISDestroy(&range.is) >> utilities::PetscUtilities::checkError;
    }
}
