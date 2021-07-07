#include "flow.hpp"
#include "utilities/mpiError.hpp"
#include "utilities/petscError.hpp"
#include "utilities/petscOptions.hpp"

ablate::flow::Flow::Flow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<parameters::Parameters> options,
                         std::vector<std::shared_ptr<mathFunctions::FieldSolution>> initialization, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                         std::vector<std::shared_ptr<mathFunctions::FieldSolution>> auxiliaryFields, std::vector<std::shared_ptr<mathFunctions::FieldSolution>> exactSolution)
    : name(name),
      dm(mesh),
      auxDM(nullptr),
      flowField(nullptr),
      auxField(nullptr),
      initialization(initialization),
      boundaryConditions(boundaryConditions),
      auxiliaryFieldsUpdaters(auxiliaryFields),
      exactSolutions(exactSolution),
      petscOptions(nullptr) {
    // Set the application context with this dm
    DMSetApplicationContext(dm->GetDomain(), this) >> checkError;

    // Get the dim and store the flows
    DMGetDimension(dm->GetDomain(), &dim) >> checkError;

    // Set the options
    if (options) {
        PetscOptionsCreate(&petscOptions) >> checkError;
        options->Fill(petscOptions);
    }

    // Register the aux fields updater if specified
    if (!auxiliaryFields.empty()) {
        this->preStepFunctions.push_back(UpdateAuxFields);
    }
}

ablate::flow::Flow::~Flow() {
    // clean up the petsc objects
    if (flowField) {
        VecDestroy(&flowField) >> checkError;
    }
    if (auxField) {
        VecDestroy(&auxField) >> checkError;
    }
    if (auxDM) {
        DMDestroy(&auxDM) >> checkError;
    }
    if (petscOptions) {
        ablate::utilities::PetscOptionsDestroyAndCheck(name, &petscOptions);
    }
}

std::optional<int> ablate::flow::Flow::GetFieldId(const std::string& fieldName) const {
    for (std::size_t f = 0; f < flowFieldDescriptors.size(); f++) {
        if (flowFieldDescriptors[f].fieldName == fieldName) {
            return f;
        }
    }
    return {};
}

std::optional<int> ablate::flow::Flow::GetAuxFieldId(const std::string& fieldName) const {
    for (std::size_t f = 0; f < auxFieldDescriptors.size(); f++) {
        if (auxFieldDescriptors[f].fieldName == fieldName) {
            return f;
        }
    }
    return {};
}

void ablate::flow::Flow::RegisterField(FlowFieldDescriptor flowFieldDescription, DM dmAdd) {
    switch (flowFieldDescription.fieldType) {
        case FieldType::FE: {
            // determine if it a simplex element and the number of dimensions
            DMPolytopeType ct;
            PetscInt cStart;
            DMPlexGetHeightStratum(dm->GetDomain(), 0, &cStart, NULL) >> checkError;
            DMPlexGetCellType(dm->GetDomain(), cStart, &ct) >> checkError;
            PetscInt simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1 ? PETSC_TRUE : PETSC_FALSE;
            PetscInt simplexGlobal;

            // Assume true if any rank says true
            MPI_Allreduce(&simplex, &simplexGlobal, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm->GetDomain())) >> checkMpiError;
            // create a petsc fe
            PetscFE petscFE;
            PetscFECreateDefault(PetscObjectComm((PetscObject)dm->GetDomain()),
                                 dim,
                                 flowFieldDescription.components,
                                 simplexGlobal ? PETSC_TRUE : PETSC_FALSE,
                                 flowFieldDescription.fieldPrefix.c_str(),
                                 PETSC_DEFAULT,
                                 &petscFE) >>
                checkError;
            PetscObjectSetName((PetscObject)petscFE, flowFieldDescription.fieldName.c_str()) >> checkError;
            PetscObjectSetOptions((PetscObject)petscFE, petscOptions) >> checkError;

            // If this is not the first field, copy the quadrature locations
            if (!flowFieldDescriptors.empty()) {
                PetscFE referencePetscFE;
                DMGetField(dm->GetDomain(), 0, NULL, (PetscObject*)&referencePetscFE) >> checkError;
                PetscFECopyQuadrature(referencePetscFE, petscFE) >> checkError;
            }

            // Store the field and destroy copy
            DMAddField(dmAdd, NULL, (PetscObject)petscFE) >> checkError;
            PetscFEDestroy(&petscFE) >> checkError;
        } break;
        case FieldType::FV: {
            PetscFV fvm;
            PetscFVCreate(PetscObjectComm((PetscObject)dm->GetDomain()), &fvm) >> checkError;
            PetscObjectSetOptionsPrefix((PetscObject)fvm, flowFieldDescription.fieldPrefix.c_str()) >> checkError;
            PetscObjectSetName((PetscObject)fvm, flowFieldDescription.fieldName.c_str()) >> checkError;
            PetscObjectSetOptions((PetscObject)fvm, petscOptions) >> checkError;

            PetscFVSetFromOptions(fvm) >> checkError;
            PetscFVSetNumComponents(fvm, flowFieldDescription.components) >> checkError;
            PetscFVSetSpatialDimension(fvm, dim) >> checkError;

            // If there are any names provided, name each component in this field this is used by some of the output fields
            for (std::size_t c = 0; c < flowFieldDescription.componentNames.size(); c++) {
                PetscFVSetComponentName(fvm, c, flowFieldDescription.componentNames[c].c_str()) >> checkError;
            }

            DMAddField(dmAdd, NULL, (PetscObject)fvm) >> checkError;
            PetscFVDestroy(&fvm) >> checkError;
        } break;
        default: {
            throw std::invalid_argument("Unknown field type for flow");
        }
    }
}

void ablate::flow::Flow::RegisterField(FlowFieldDescriptor flowFieldDescription) {
    // add solution fields/aux fields
    if (flowFieldDescription.solutionField) {
        // Called the shared method to register
        RegisterField(flowFieldDescription, dm->GetDomain());

        // store the field
        flowFieldDescriptors.push_back(flowFieldDescription);
    } else {
        // check to see if need to create an aux dm
        if (auxDM == NULL) {
            /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
            DM coordDM;
            DMGetCoordinateDM(dm->GetDomain(), &coordDM) >> checkError;
            DMClone(dm->GetDomain(), &auxDM) >> checkError;

            // this is a hard coded "dmAux" that petsc looks for
            PetscObjectCompose((PetscObject)dm->GetDomain(), "dmAux", (PetscObject)auxDM) >> checkError;
            DMSetCoordinateDM(auxDM, coordDM) >> checkError;
        }

        RegisterField(flowFieldDescription, auxDM);

        // store the field
        auxFieldDescriptors.push_back(flowFieldDescription);
    }
}

void ablate::flow::Flow::FinalizeRegisterFields() { DMCreateDS(dm->GetDomain()) >> checkError; }

PetscErrorCode ablate::flow::Flow::TSPreStepFunction(TS ts) {
    PetscFunctionBeginUser;
    DM dm;
    PetscErrorCode ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ablate::flow::Flow* flowObject;
    ierr = DMGetApplicationContext(dm, &flowObject);
    CHKERRQ(ierr);

    for (const auto& function : flowObject->preStepFunctions) {
        try {
            function(ts, *flowObject);
        } catch (std::exception& exp) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exp.what());
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::flow::Flow::TSPostStepFunction(TS ts) {
    PetscFunctionBeginUser;
    DM dm;
    PetscErrorCode ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ablate::flow::Flow* flowObject;
    ierr = DMGetApplicationContext(dm, &flowObject);
    CHKERRQ(ierr);

    for (const auto& function : flowObject->postStepFunctions) {
        try {
            function(ts, *flowObject);
        } catch (std::exception& exp) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exp.what());
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::flow::Flow::TSPostEvaluateFunction(TS ts) {
    PetscFunctionBeginUser;
    DM dm;
    PetscErrorCode ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ablate::flow::Flow* flowObject;
    ierr = DMGetApplicationContext(dm, &flowObject);
    CHKERRQ(ierr);

    for (const auto& function : flowObject->postEvaluateFunctions) {
        try {
            function(ts, *flowObject);
        } catch (std::exception& exp) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exp.what());
        }
    }

    PetscFunctionReturn(0);
}

void ablate::flow::Flow::CompleteProblemSetup(TS ts) {
    // Apply any boundary conditions
    PetscDS prob;
    DMGetDS(dm->GetDomain(), &prob) >> checkError;

    // add each boundary condition
    for (auto boundary : boundaryConditions) {
        auto fieldId = GetFieldId(boundary->GetFieldName());
        if (!fieldId) {
            throw std::invalid_argument("unknown field for boundary: " + boundary->GetFieldName());
        }

        // Setup the boundary condition
        boundary->SetupBoundary(prob, fieldId.value());
    }

    // Setup the solve with the ts
    TSSetDM(ts, dm->GetDomain()) >> checkError;
    DMPlexCreateClosureIndex(dm->GetDomain(), NULL) >> checkError;
    DMCreateGlobalVector(dm->GetDomain(), &(flowField)) >> checkError;
    PetscObjectSetName((PetscObject)flowField, "flowField") >> checkError;

    if (auxDM) {
        DMCreateDS(auxDM) >> checkError;
        DMCreateLocalVector(auxDM, &(auxField)) >> checkError;

        // attach this field as aux vector to the dm
        PetscObjectCompose((PetscObject)dm->GetDomain(), "A", (PetscObject)auxField) >> checkError;
        PetscObjectSetName((PetscObject)auxField, "auxField") >> checkError;
    }

    // Check if any of the fields are fe
    PetscBool isFE = PETSC_FALSE;
    PetscBool isFV = PETSC_FALSE;
    for (const auto& flowField : flowFieldDescriptors) {
        switch (flowField.fieldType) {
            case (FieldType::FE):
                isFE = PETSC_TRUE;
                break;
            case (FieldType::FV):
                isFV = PETSC_TRUE;
                break;
        }
    }

    if (isFE) {
        DMTSSetBoundaryLocal(dm->GetDomain(), DMPlexTSComputeBoundary, NULL) >> checkError;
        DMTSSetIFunctionLocal(dm->GetDomain(), DMPlexTSComputeIFunctionFEM, NULL) >> checkError;
        DMTSSetIJacobianLocal(dm->GetDomain(), DMPlexTSComputeIJacobianFEM, NULL) >> checkError;
    }
    if (isFV) {
        DMTSSetRHSFunctionLocal(dm->GetDomain(), DMPlexTSComputeRHSFunctionFVM, this) >> checkError;
    }
    TSSetPreStep(ts, TSPreStepFunction) >> checkError;
    TSSetPostStep(ts, TSPostStepFunction) >> checkError;
    TSSetPostEvaluate(ts, TSPostEvaluateFunction) >> checkError;

    // Initialize the flow field if provided
    if (!initialization.empty()) {
        PetscInt numberFields;
        DMGetNumFields(dm->GetDomain(), &numberFields) >> checkError;

        // size up the update and context functions
        std::vector<mathFunctions::PetscFunction> fieldFunctions(numberFields, NULL);
        std::vector<void*> fieldContexts(numberFields, NULL);

        for (auto fieldInitialization : initialization) {
            auto fieldId = GetFieldId(fieldInitialization->GetName());
            if (!fieldId) {
                throw std::invalid_argument("unknown field for initialization: " + fieldInitialization->GetName());
            }

            fieldContexts[fieldId.value()] = fieldInitialization->GetSolutionField().GetContext();
            fieldFunctions[fieldId.value()] = fieldInitialization->GetSolutionField().GetPetscFunction();
        }

        DMProjectFunction(dm->GetDomain(), 0.0, &fieldFunctions[0], &fieldContexts[0], INSERT_VALUES, flowField) >> checkError;
        this->CompleteFlowInitialization(dm->GetDomain(), flowField);
    }

    // if an exact solution has been provided register it
    for (const auto& exactSolution : exactSolutions) {
        auto fieldId = GetFieldId(exactSolution->GetName());
        if (!fieldId) {
            throw std::invalid_argument("unknown field for exact solution: " + exactSolution->GetName());
        }

        // Get the current field type
        if (exactSolution->HasSolutionField()) {
            PetscDSSetExactSolution(prob, fieldId.value(), exactSolution->GetSolutionField().GetPetscFunction(), exactSolution->GetSolutionField().GetContext()) >> checkError;
        }
        if (exactSolution->HasTimeDerivative()) {
            PetscDSSetExactSolutionTimeDerivative(prob, fieldId.value(), exactSolution->GetTimeDerivative().GetPetscFunction(), exactSolution->GetTimeDerivative().GetContext()) >> checkError;
        }
    }
}

/**
 * Static function that is called by the flow object to update the aux variables if an aux variable soltuion was provided
 * @param ts
 * @param flow
 */
void ablate::flow::Flow::UpdateAuxFields(TS ts, ablate::flow::Flow& flow) {
    PetscInt numberAuxFields;
    DMGetNumFields(flow.auxDM, &numberAuxFields) >> checkError;

    // size up the update and context functions
    std::vector<mathFunctions::PetscFunction> auxiliaryFieldFunctions(numberAuxFields, NULL);
    std::vector<void*> auxiliaryFieldContexts(numberAuxFields, NULL);

    // for each given aux field
    for (auto auxFieldDescription : flow.auxiliaryFieldsUpdaters) {
        auto fieldId = flow.GetAuxFieldId(auxFieldDescription->GetName());
        if (!fieldId) {
            throw std::invalid_argument("unknown field for aux field: " + auxFieldDescription->GetName());
        }

        auxiliaryFieldContexts[fieldId.value()] = auxFieldDescription->GetSolutionField().GetContext();
        auxiliaryFieldFunctions[fieldId.value()] = auxFieldDescription->GetSolutionField().GetPetscFunction();
    }

    // get the time at the end of the time step
    PetscReal time = 0;
    PetscReal dt = 0;
    TSGetTime(ts, &time) >> checkError;
    TSGetTimeStep(ts, &dt) >> checkError;

    // Update the source terms
    DMProjectFunctionLocal(flow.auxDM, time + dt, &auxiliaryFieldFunctions[0], &auxiliaryFieldContexts[0], INSERT_ALL_VALUES, flow.auxField) >> checkError;
}

void ablate::flow::Flow::View(PetscViewer viewer, PetscInt steps, PetscReal time, Vec u) const {
    // If this is the first output, save the mesh
    if (steps == 0) {
        // Print the initial mesh
        DMView(GetDM(), viewer) >> checkError;

        if (auxDM) {
            DMSetOutputSequenceNumber(auxDM, steps, time) >> checkError;
        }
    }

    // Always save the main flowField
    VecView(flowField, viewer) >> checkError;

    // If there is aux data output
    if (auxField) {
        // copy over the sequence data from the main dm
        PetscReal dmTime;
        PetscInt dmSequence;
        DMGetOutputSequenceNumber(GetDM(), &dmSequence, &dmTime) >> checkError;
        DMSetOutputSequenceNumber(auxDM, dmSequence, dmTime) >> checkError;

        Vec auxGlobalField;
        DMGetGlobalVector(auxDM, &auxGlobalField) >> checkError;

        // copy over the name of the auxFieldVector
        const char* name;
        PetscObjectGetName((PetscObject)auxField, &name) >> checkError;
        PetscObjectSetName((PetscObject)auxGlobalField, name) >> checkError;
        DMLocalToGlobal(auxDM, auxField, INSERT_VALUES, auxGlobalField) >> checkError;
        VecView(auxGlobalField, viewer) >> checkError;
        DMRestoreGlobalVector(auxDM, &auxGlobalField) >> checkError;
    }

    if (!exactSolutions.empty()) {
        Vec exactVec;
        DMGetGlobalVector(dm->GetDomain(), &exactVec) >> checkError;

        // Get the number of fields
        PetscDS ds;
        DMGetDS(dm->GetDomain(), &ds) >> checkError;
        PetscInt numberOfFields;
        PetscDSGetNumFields(ds, &numberOfFields) >> checkError;
        std::vector<ablate::mathFunctions::PetscFunction> exactFuncs(numberOfFields);
        std::vector<void*> exactCtxs(numberOfFields);
        for (auto f = 0; f < numberOfFields; ++f) {
            PetscDSGetExactSolution(ds, f, &exactFuncs[f], &exactCtxs[f]) >> checkError;
            if (!exactFuncs[f]) {
                throw std::invalid_argument("The exact solution has not set");
            }
        }

        DMProjectFunction(dm->GetDomain(), time, &exactFuncs[0], &exactCtxs[0], INSERT_ALL_VALUES, exactVec) >> checkError;

        PetscObjectSetName((PetscObject)exactVec, "exact") >> checkError;
        VecView(exactVec, viewer) >> checkError;
        DMRestoreGlobalVector(dm->GetDomain(), &exactVec) >> checkError;
    }
}
