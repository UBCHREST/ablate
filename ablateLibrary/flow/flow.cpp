#include "flow.hpp"
#include "../utilities/petscError.hpp"
#include "../utilities/petscOptions.hpp"
#include "utilities/mpiError.hpp"

ablate::flow::Flow::Flow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<parameters::Parameters> options,
                         std::vector<std::shared_ptr<FlowFieldSolution>> initialization, std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions,
                         std::vector<std::shared_ptr<FlowFieldSolution>> auxiliaryFields)
    : name(name), dm(nullptr), auxDM(nullptr), flowField(nullptr), auxField(nullptr), petscOptions(nullptr), initialization(initialization), boundaryConditions(boundaryConditions), auxiliaryFields(auxiliaryFields) {
    // Copy the dm and set the value
    dm = mesh->GetDomain();

    // Set the application context with this dm
    DMSetApplicationContext(dm, this) >> checkError;

    // Get the dim and store the flows
    DMGetDimension(dm, &dim) >> checkError;

    // Set the options
    if(options){
        PetscOptionsCreate(&petscOptions)  >> checkError;
        options->Fill(petscOptions);
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
    if (dm) {
        DMDestroy(&dm) >> checkError;
    }
    if (auxDM) {
        DMDestroy(&auxDM) >> checkError;
    }
    if(petscOptions){
        PetscOptionsDestroy(&petscOptions) >> checkError;
    }
}

void ablate::flow::Flow::RegisterField(FlowFieldDescriptor flowFieldDescription) {
    // store the field
    flowFieldDescriptors.push_back(flowFieldDescription);

    switch (flowFieldDescription.fieldType) {
        case FieldType::FE: {
            // determine if it a simplex element and the number of dimensions
            DMPolytopeType ct;
            PetscInt cStart;
            DMPlexGetHeightStratum(dm, 0, &cStart, NULL) >> checkError;
            DMPlexGetCellType(dm, cStart, &ct) >> checkError;
            PetscInt simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1 ? PETSC_TRUE : PETSC_FALSE;
            PetscInt simplexGlobal;

            // Assume true if any rank says true
            MPI_Allreduce(&simplex, &simplexGlobal, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm)) >> checkMpiError;

            // create a petsc fe
            PetscFE petscFE;
            PetscFECreateDefault(
                PetscObjectComm((PetscObject)dm), dim, flowFieldDescription.components, simplexGlobal ? PETSC_TRUE : PETSC_FALSE, flowFieldDescription.fieldPrefix.c_str(), PETSC_DEFAULT, &petscFE) >>
                checkMpiError;
            PetscObjectSetName((PetscObject)petscFE, flowFieldDescription.fieldName.c_str()) >> checkError;
            PetscObjectSetOptions((PetscObject)petscFE, petscOptions) >> checkError;  // TODO: update with options

            // If this is not the first field, copy the quadrature locations
            if (flowFieldDescriptors.size() > 1) {
                PetscFE referencePetscFE;
                DMGetField(dm, 0, NULL, (PetscObject*)&referencePetscFE) >> checkError;
                PetscFECopyQuadrature(referencePetscFE, petscFE) >> checkError;
            }

            // Store the field and destroy copy
            DMAddField(dm, NULL, (PetscObject)petscFE) >> checkError;
            PetscFEDestroy(&petscFE) >> checkError;
        } break;
        case FieldType::FV: {
            PetscFV fvm;
            PetscFVCreate(PETSC_COMM_WORLD, &fvm) >> checkError;
            PetscObjectSetOptionsPrefix((PetscObject)fvm, flowFieldDescription.fieldPrefix.c_str()) >> checkError;
            PetscObjectSetName((PetscObject)fvm, flowFieldDescription.fieldName.c_str()) >> checkError;
            PetscObjectSetOptions((PetscObject)fvm, petscOptions) >> checkError;  // TODO: update with options

            PetscFVSetFromOptions(fvm) >> checkError;
            PetscFVSetNumComponents(fvm, flowFieldDescription.components) >> checkError;
            PetscFVSetSpatialDimension(fvm, dim) >> checkError;

            DMAddField(dm, NULL, (PetscObject)fvm) >> checkError;
            PetscFVDestroy(&fvm) >> checkError;
        } break;
        default: {
            throw std::invalid_argument("Unknown field type for flow");
        }
    }
}

void ablate::flow::Flow::FinalizeRegisterFields() { DMCreateDS(dm) >> checkError; }

void ablate::flow::Flow::RegisterAuxField(FlowFieldDescriptor flowFieldDescription) {
    // store the field
    auxFieldDescriptors.push_back(flowFieldDescription);

    // check to see if need to create an aux dm
    if (auxDM == NULL) {
        /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
        DM coordDM;
        DMGetCoordinateDM(dm, &coordDM) >> checkError;
        DMClone(dm, &auxDM) >> checkError;

        // this is a hard coded "dmAux" that petsc looks for
        PetscObjectCompose((PetscObject)dm, "dmAux", (PetscObject)auxDM) >> checkError;
        DMSetCoordinateDM(auxDM, coordDM) >> checkError;
    }

    switch (flowFieldDescription.fieldType) {
        case FieldType::FE: {
            // determine if it a simplex element and the number of dimensions
            DMPolytopeType ct;
            PetscInt cStart;
            DMPlexGetHeightStratum(dm, 0, &cStart, NULL) >> checkError;
            DMPlexGetCellType(dm, cStart, &ct) >> checkError;
            PetscInt simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1 ? PETSC_TRUE : PETSC_FALSE;
            PetscInt simplexGlobal;

            // Assume true if any rank says true
            MPI_Allreduce(&simplex, &simplexGlobal, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm)) >> checkMpiError;
            // create a petsc fe
            PetscFE petscFE;
            PetscFECreateDefault(
                PetscObjectComm((PetscObject)dm), dim, flowFieldDescription.components, simplexGlobal ? PETSC_TRUE : PETSC_FALSE, flowFieldDescription.fieldPrefix.c_str(), PETSC_DEFAULT, &petscFE) >>
                checkError;
            PetscObjectSetName((PetscObject)petscFE, flowFieldDescription.fieldName.c_str()) >> checkError;
            PetscObjectSetOptions((PetscObject)petscFE, petscOptions) >> checkError;  // TODO: update with options

            // If this is not the first field, copy the quadrature locations
            if (!flowFieldDescriptors.empty()) {
                PetscFE referencePetscFE;
                DMGetField(dm, 0, NULL, (PetscObject*)&referencePetscFE) >> checkError;
                PetscFECopyQuadrature(referencePetscFE, petscFE) >> checkError;
            }

            // Store the field and destroy copy
            DMAddField(auxDM, NULL, (PetscObject)petscFE) >> checkError;
            PetscFEDestroy(&petscFE) >> checkError;
        } break;
        case FieldType::FV: {
            PetscFV fvm;
            PetscFVCreate(PETSC_COMM_WORLD, &fvm) >> checkError;
            PetscObjectSetOptionsPrefix((PetscObject)fvm, flowFieldDescription.fieldPrefix.c_str()) >> checkError;
            PetscObjectSetName((PetscObject)fvm, flowFieldDescription.fieldName.c_str()) >> checkError;
            PetscObjectSetOptions((PetscObject)fvm, petscOptions) >> checkError;  // TODO: update with options

            PetscFVSetFromOptions(fvm) >> checkError;
            PetscFVSetNumComponents(fvm, flowFieldDescription.components) >> checkError;
            PetscFVSetSpatialDimension(fvm, dim) >> checkError;

            DMAddField(auxDM, NULL, (PetscObject)fvm) >> checkError;
            PetscFVDestroy(&fvm) >> checkError;
        } break;
        default: {
            throw std::invalid_argument("Unknown field type for flow");
        }
    }
}
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

void ablate::flow::Flow::CompleteProblemSetup(TS ts) {
    // Setup the solve with the ts
    TSSetDM(ts, dm) >> checkError;
    DMPlexCreateClosureIndex(dm, NULL) >> checkError;
    DMCreateGlobalVector(dm, &(flowField)) >> checkError;

    if (auxDM) {
        DMCreateDS(auxDM) >> checkError;
        DMCreateLocalVector(auxDM, &(auxField)) >> checkError;

        // attach this field as aux vector to the dm
        PetscObjectCompose((PetscObject)dm, "A", (PetscObject)auxField) >> checkError;
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
        DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, NULL)  >> checkError;
        DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, NULL)  >> checkError;
        DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, NULL) >> checkError;
    }
    if (isFV) {
        DMTSSetRHSFunctionLocal(dm, DMPlexTSComputeRHSFunctionFVM, this) >> checkError;
    }
    TSSetPreStep(ts, TSPreStepFunction) >> checkError;
    TSSetPostStep(ts, TSPostStepFunction) >> checkError;
}
