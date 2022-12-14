#include "subDomain.hpp"
#include <set>
#include <sstream>
#include <utilities/petscError.hpp>
#include "utilities/mpiError.hpp"

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

            // Get the offset for this field
            PetscInt fieldOffset;
            PetscDSGetFieldOffset(discreteSystem, f, &fieldOffset) >> checkError;

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
            PetscDSGetFieldOffset(discreteSystem, field.id, &fieldOffset) >> checkError;

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
        DMGetCoordinateDM(domain.GetDM(), &coordDM) >> checkError;
        DMClone(domain.GetDM(), &auxDM) >> checkError;

        // this is a hard coded "dmAux" that petsc looks for
        DMSetCoordinateDM(auxDM, coordDM) >> checkError;

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
        DMDestroy(&auxDM) >> checkError;
    }

    if (auxLocalVec) {
        VecDestroy(&auxLocalVec) >> checkError;
    }

    if (auxGlobalVec) {
        VecDestroy(&auxGlobalVec) >> checkError;
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

void ablate::domain::SubDomain::CreateSubDomainStructures() {
    if (auxDM) {
        DMCreateDS(auxDM) >> checkError;
        DMCreateLocalVector(auxDM, &(auxLocalVec)) >> checkError;
        DMCreateGlobalVector(auxDM, &(auxGlobalVec)) >> checkError;

        DMGetRegionDS(auxDM, label, nullptr, &auxDiscreteSystem) >> checkError;

        // attach this field as aux vector to the dm
        DMSetAuxiliaryVec(domain.GetDM(), label, labelValue, 0 /*The equation part, or 0 if unused*/, auxLocalVec) >> checkError;
        PetscObjectSetName((PetscObject)auxLocalVec, "aux") >> checkError;
        PetscObjectSetName((PetscObject)auxGlobalVec, "aux") >> checkError;

        // add the names to each of the components in the dm section
        PetscSection section;
        DMGetLocalSection(auxDM, &section) >> checkError;
        for (const auto& field : GetFields(FieldLocation::AUX)) {
            if (field.numberComponents > 1) {
                for (PetscInt c = 0; c < field.numberComponents; c++) {
                    PetscSectionSetComponentName(section, field.id, c, field.components[c].c_str()) >> checkError;
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

    VecGetDM(globVec, &dm) >> checkError;
    if (dm != subDM) {
        throw std::invalid_argument("The Vector passed in to ablate::domain::SubDomain::ProjectFieldFunctionsToSubDM must be a global vector from the SubDM.");
    }
    DMGetNumFields(subDM, &numberFields) >> checkError;

    // get a local vector for the work
    Vec locVec;
    DMGetLocalVector(dm, &locVec) >> checkError;
    DMGlobalToLocal(dm, globVec, INSERT_VALUES, locVec) >> checkError;

    // size up the update and context functions
    std::vector<mathFunctions::PetscFunction> fieldFunctions(numberFields, nullptr);
    std::vector<void*> fieldContexts(numberFields, nullptr);

    for (auto& fieldInitialization : initialization) {
        auto fieldId = GetField(fieldInitialization->GetName());

        fieldContexts[fieldId.subId] = fieldInitialization->GetSolutionField().GetContext();
        fieldFunctions[fieldId.subId] = fieldInitialization->GetSolutionField().GetPetscFunction();
    }

    DMProjectFunctionLocal(subDM, time, &fieldFunctions[0], &fieldContexts[0], INSERT_VALUES, locVec) >> checkError;

    // push the results back to the global vector
    DMLocalToGlobal(dm, locVec, INSERT_VALUES, globVec) >> checkError;
    DMRestoreLocalVector(dm, &locVec) >> checkError;
}

Vec ablate::domain::SubDomain::GetAuxGlobalVector() {
    if (!auxDM) {
        return nullptr;
    }
    DMLocalToGlobal(auxDM, auxLocalVec, INSERT_VALUES, auxGlobalVec) >> checkError;
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
        DMPlexFilter(GetDM(), label, labelValue, &subDM) >> checkError;

        // copy over all fields that were in the main dm
        for (auto& fieldInfo : GetFields()) {
            auto petscField = GetPetscFieldObject(fieldInfo);
            DMAddField(subDM, nullptr, petscField) >> checkError;
        }

        // add the names to each of the components in the dm section
        PetscSection section;
        DMGetLocalSection(subDM, &section) >> checkError;
        for (const auto& field : GetFields()) {
            if (field.numberComponents > 1) {
                for (PetscInt c = 0; c < field.numberComponents; c++) {
                    PetscSectionSetComponentName(section, field.id, c, field.components[c].c_str()) >> checkError;
                }
            }
        }

        DMCreateDS(subDM) >> checkError;

        // Copy over options
        PetscOptions options;
        PetscObjectGetOptions((PetscObject)GetDM(), &options) >> checkError;
        PetscObjectSetOptions((PetscObject)subDM, options) >> checkError;
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

    // Make a local version of the vector
    Vec localSolutionVec;
    DMGetLocalVector(GetDM(), &localSolutionVec) >> checkError;
    DMGlobalToLocalBegin(GetDM(), GetSolutionVector(), INSERT_VALUES, localSolutionVec) >> checkError;
    DMGlobalToLocalEnd(GetDM(), GetSolutionVector(), INSERT_VALUES, localSolutionVec) >> checkError;

    CopyGlobalToSubVector(subDM, GetDM(), subSolutionVec, localSolutionVec, GetFields(), GetFields(), true);
    DMRestoreLocalVector(GetDM(), &localSolutionVec) >> checkError;
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

    // add the names to each of the components in the dm section
    PetscSection section;
    DMGetLocalSection(subAuxDM, &section) >> checkError;
    for (const auto& field : GetFields(FieldLocation::AUX)) {
        if (field.numberComponents > 1) {
            for (PetscInt c = 0; c < field.numberComponents; c++) {
                PetscSectionSetComponentName(section, field.id, c, field.components[c].c_str()) >> checkError;
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
    DMGetLocalSection(sDM, &section) >> checkError;

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
            PetscScalar* subRef = nullptr;
            const PetscScalar* ref = nullptr;

            DMPlexPointGlobalFieldRef(sDM, p, subFieldInfo.id, subVecArray, &subRef) >> checkError;
            if (localVector) {
                DMPlexPointLocalFieldRead(gDM, gP, globFieldInfo.id, globalVecArray, &ref) >> checkError;
            } else {
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

void ablate::domain::SubDomain::CopySubVectorToGlobal(DM sDM, DM gDM, Vec subVec, Vec globVec, const std::vector<Field>& subFields, const std::vector<Field>& gFields, bool localVector) const {
    /* Get the map from the subVec to global */
    IS subpointIS;
    const PetscInt* subpointIndices = nullptr;
    DMPlexGetSubpointIS(sDM, &subpointIS) >> checkError;
    ISGetIndices(subpointIS, &subpointIndices) >> checkError;

    // Get array access to the vec
    PetscScalar* globalVecArray;
    const PetscScalar* subVecArray;
    VecGetArray(globVec, &globalVecArray) >> checkError;
    VecGetArrayRead(subVec, &subVecArray) >> checkError;

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
            const PetscScalar* subRef = nullptr;
            PetscScalar* ref = nullptr;

            DMPlexPointGlobalFieldRead(sDM, p, subFieldInfo.id, subVecArray, &subRef) >> checkError;
            if (localVector) {
                DMPlexPointLocalFieldRef(gDM, gP, globFieldInfo.id, globalVecArray, &ref) >> checkError;
            } else {
                DMPlexPointGlobalFieldRef(gDM, gP, globFieldInfo.id, globalVecArray, &ref) >> checkError;
            }

            if (subRef && ref) {
                // Copy the point
                PetscArraycpy(ref, subRef, numberComponents) >> checkError;
            }
        }
    }
    VecRestoreArray(globVec, &globalVecArray) >> checkError;
    VecRestoreArrayRead(subVec, &subVecArray) >> checkError;
    ISRestoreIndices(subpointIS, &subpointIndices) >> checkError;
}

bool ablate::domain::SubDomain::InRegion(const domain::Region& region) const {
    if (!label) {
        return true;
    }

    // Compute the size of region inside the subDomain label
    PetscInt size;

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

    // each rank must check separately and then share the result
    if (regionIS) {
        // Check for an overlap
        IS overlapIS;
        ISIntersect(dsLabelIS, regionIS, &overlapIS) >> checkError;

        // determine if there is an overlap
        if (overlapIS) {
            ISGetSize(overlapIS, &size) >> checkError;
            ISDestroy(&overlapIS) >> checkError;
        }
        ISDestroy(&regionIS) >> checkError;
    }
    ISDestroy(&dsLabelIS) >> checkError;

    // Take the sum
    PetscInt globalSize;
    MPI_Allreduce(&size, &globalSize, 1, MPIU_INT, MPI_SUM, GetComm()) >> checkMpiError;
    return globalSize > 0;
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

PetscErrorCode ablate::domain::SubDomain::GetFieldGlobalVector(const Field& field, IS* vecIs, Vec* vec, DM* subdm) {
    PetscFunctionBeginUser;
    // Get the correct dm
    auto entireDm = GetFieldDM(field);
    auto entireVec = GetGlobalVec(field);

    PetscErrorCode ierr;
    ierr = DMCreateSubDM(entireDm, 1, &field.id, vecIs, subdm);
    CHKERRQ(ierr);

    // Get the sub vector
    ierr = VecGetSubVector(entireVec, *vecIs, vec);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::domain::SubDomain::RestoreFieldGlobalVector(const Field& field, IS* vecIs, Vec* vec, DM* subdm) {
    PetscFunctionBeginUser;
    auto entireVec = GetGlobalVec(field);
    PetscErrorCode ierr;
    ierr = VecRestoreSubVector(entireVec, *vecIs, vec);
    CHKERRQ(ierr);
    ierr = ISDestroy(vecIs);
    CHKERRQ(ierr);
    ierr = DMDestroy(subdm);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::domain::SubDomain::GetFieldLocalVector(const ablate::domain::Field& field, PetscReal time, IS* vecIs, Vec* vec, DM* subdm) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    if (field.location == FieldLocation::SOL) {
        // Get the correct dm
        auto entireDm = GetDM();
        auto entireVec = GetSolutionVector();

        // Create a subDM
        ierr = DMCreateSubDM(entireDm, 1, &field.id, vecIs, subdm);
        CHKERRQ(ierr);

        // Use a global vector to get the results
        Vec subGlobalVector;
        ierr = VecGetSubVector(entireVec, *vecIs, &subGlobalVector);
        CHKERRQ(ierr);

        // Make a local version of the vector
        ierr = DMGetLocalVector(*subdm, vec);
        CHKERRQ(ierr);
        ierr = DMPlexInsertBoundaryValues(*subdm, PETSC_TRUE, *vec, time, nullptr, nullptr, nullptr);
        CHKERRQ(ierr);
        ierr = DMGlobalToLocalBegin(*subdm, subGlobalVector, INSERT_VALUES, *vec);
        CHKERRQ(ierr);
        ierr = DMGlobalToLocalEnd(*subdm, subGlobalVector, INSERT_VALUES, *vec);
        CHKERRQ(ierr);

        // We have the filled local vec subdm, so clean up the subGlobalVector and vecIS
        ierr = VecRestoreSubVector(entireVec, *vecIs, &subGlobalVector);
        CHKERRQ(ierr);
        ierr = ISDestroy(vecIs);
        *vecIs = nullptr;
        CHKERRQ(ierr);
    } else if (field.location == FieldLocation::AUX) {
        auto entireDm = GetAuxDM();
        auto entireVec = GetAuxVector();

        ierr = DMCreateSubDM(entireDm, 1, &field.id, vecIs, subdm);
        CHKERRQ(ierr);

        // Get the sub vector
        ierr = VecGetSubVector(entireVec, *vecIs, vec);
        CHKERRQ(ierr);
    } else {
        SETERRQ(GetComm(), PETSC_ERR_SUP, "%s", "Unknown field location");
    }

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::domain::SubDomain::RestoreFieldLocalVector(const ablate::domain::Field& field, IS* vecIs, Vec* vec, DM* subdm) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    if (field.location == FieldLocation::SOL) {
        // In the Get call, the vecIS was already cleaned up and vec is only a localVec
        ierr = DMRestoreLocalVector(*subdm, vec);
        CHKERRQ(ierr);
        ierr = DMDestroy(subdm);
        CHKERRQ(ierr);
    } else if (field.location == FieldLocation::AUX) {
        auto entireVec = GetAuxVector();
        ierr = VecRestoreSubVector(entireVec, *vecIs, vec);
        CHKERRQ(ierr);
        ierr = ISDestroy(vecIs);
        CHKERRQ(ierr);
        ierr = DMDestroy(subdm);
        CHKERRQ(ierr);
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
                PetscDSSetExactSolution(GetDiscreteSystem(), fieldInfo.subId, exactSolution->GetSolutionField().GetPetscFunction(), exactSolution->GetSolutionField().GetContext()) >> checkError;
            }
            if (exactSolution->HasTimeDerivative()) {
                PetscDSSetExactSolutionTimeDerivative(GetDiscreteSystem(), fieldInfo.subId, exactSolution->GetTimeDerivative().GetPetscFunction(), exactSolution->GetTimeDerivative().GetContext()) >>
                    checkError;
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
        DMView(locSubDm, viewer) >> checkError;
    }

    // set the dm sequence number, because we may be skipping outputs
    DMSetOutputSequenceNumber(locSubDm, sequenceNumber, time) >> checkError;
    if (locAuxDM) {
        DMSetOutputSequenceNumber(locAuxDM, sequenceNumber, time) >> checkError;
    }

    // Always save the main flowField
    VecView(GetSubSolutionVector(), viewer) >> checkError;

    // If there is aux data output
    PetscBool outputAuxVector = PETSC_TRUE;
    PetscOptionsGetBool(nullptr, nullptr, "-outputAuxVector", &outputAuxVector, nullptr) >> checkError;
    if (outputAuxVector) {
        if (auto subAuxVector = GetSubAuxVector()) {
            // copy over the sequence data from the main dm
            PetscReal dmTime;
            PetscInt dmSequence;
            DMGetOutputSequenceNumber(locSubDm, &dmSequence, &dmTime) >> checkError;
            DMSetOutputSequenceNumber(locAuxDM, dmSequence, dmTime) >> checkError;

            VecView(subAuxVector, viewer) >> checkError;
        }
    }

    // If there is an exact solution save it
    if (!exactSolutions.empty()) {
        Vec exactVec;
        DMGetGlobalVector(GetSubDM(), &exactVec) >> checkError;

        ProjectFieldFunctionsToSubDM(exactSolutions, exactVec, time);

        PetscObjectSetName((PetscObject)exactVec, "exact") >> checkError;
        VecView(exactVec, viewer) >> checkError;
        DMRestoreGlobalVector(GetSubDM(), &exactVec) >> checkError;
    }
    PetscFunctionReturnVoid();
}
void ablate::domain::SubDomain::Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    // The only item that needs to be explicitly restored is the flowField
    DMSetOutputSequenceNumber(GetDM(), sequenceNumber, time) >> checkError;
    DMSetOutputSequenceNumber(GetSubDM(), sequenceNumber, time) >> checkError;
    auto solutionVector = GetSubSolutionVector();
    VecLoad(solutionVector, viewer) >> checkError;

    // copy back if needed
    if (subSolutionVec == solutionVector) {
        CopySubVectorToGlobal(subDM, GetDM(), subSolutionVec, GetSolutionVector(), GetFields());
    }
}
void ablate::domain::SubDomain::ProjectFieldFunctionsToLocalVector(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& fieldFunctions, Vec locVec, PetscReal time) const {
    PetscInt numberFields;
    DM dm;

    VecGetDM(locVec, &dm) >> checkError;
    DMGetNumFields(dm, &numberFields) >> checkError;

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
            DMProjectFunctionLabelLocal(dm, time, fieldLabel, 1, &fieldValue, -1, nullptr, fieldFunctionsPts.data(), fieldContexts.data(), INSERT_VALUES, locVec) >> checkError;
        } else {
            DMProjectFunctionLocal(dm, time, fieldFunctionsPts.data(), fieldContexts.data(), INSERT_VALUES, locVec) >> checkError;
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
