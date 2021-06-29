#include "fvFlow.hpp"
#include <utilities/petscError.hpp>
ablate::flow::FVFlow::FVFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<parameters::Parameters> options,
                             std::vector<std::shared_ptr<mathFunctions::FieldSolution>> initialization, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                             std::vector<std::shared_ptr<mathFunctions::FieldSolution>> auxiliaryFields, std::vector<std::shared_ptr<mathFunctions::FieldSolution>> exactSolution)
    : Flow(name, mesh, parameters, options, initialization, boundaryConditions, auxiliaryFields, exactSolution) {}

PetscErrorCode ablate::flow::FVFlow::FVRHSFunctionLocal(DM dm, PetscReal time, Vec locXVec, Vec globFVec, void* ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    ablate::flow::FVFlow* flow = (ablate::flow::FVFlow*)ctx;

    /* Handle non-essential (e.g. outflow) boundary values.  This should be done before the auxFields are updated so that boundary values can be updated */
    Vec facegeom, cellgeom;
    ierr = DMPlexGetGeometryFVM(dm, &facegeom, &cellgeom, NULL);
    CHKERRQ(ierr);
    ierr = DMPlexInsertBoundaryValues(dm, PETSC_FALSE, locXVec, time, facegeom, cellgeom, NULL);
    CHKERRQ(ierr);

    // update any aux fields, including ghost cells
    ierr = FVFlowUpdateAuxFieldsFV(
        flow->dm->GetDomain(), flow->auxDM, time, locXVec, flow->auxField, flow->auxFieldUpdateFunctions.size(), &flow->auxFieldUpdateFunctions[0], &flow->auxFieldUpdateContexts[0]);
    CHKERRQ(ierr);

    // compute the  flux across each face and point wise functions(note CompressibleFlowComputeEulerFlux has already been registered)
    ierr = ABLATE_DMPlexComputeRHSFunctionFVM(&flow->rhsFluxFunctionDescriptions[0], flow->rhsFluxFunctionDescriptions.size(), &flow->rhsPointFunctionDescriptions[0], flow->rhsPointFunctionDescriptions.size(), dm, time, locXVec, globFVec);
    CHKERRQ(ierr);

    // iterate over any arbitrary RHS functions
    for(const auto& rhsFunction: flow->rhsArbitraryFunctions){
        ierr = rhsFunction.first(dm, time, locXVec, globFVec, rhsFunction.second);
        CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}
void ablate::flow::FVFlow::CompleteProblemSetup(TS ts) {
    Flow::CompleteProblemSetup(ts);

    // Override the DMTSSetRHSFunctionLocal in DMPlexTSComputeRHSFunctionFVM with a function that includes euler and diffusion source terms
    DMTSSetRHSFunctionLocal(dm->GetDomain(), FVRHSFunctionLocal, this) >> checkError;

    // copy over any boundary information from the dm, to the aux dm and set the sideset
    if (auxDM) {
        PetscDS flowProblem;
        DMGetDS(dm->GetDomain(), &flowProblem) >> checkError;
        PetscDS auxProblem;
        DMGetDS(auxDM, &auxProblem) >> checkError;

        // Get the number of boundary conditions and other info
        PetscInt numberBC;
        PetscDSGetNumBoundary(flowProblem, &numberBC) >> checkError;
        PetscInt numberAuxFields;
        PetscDSGetNumFields(auxProblem, &numberAuxFields) >> checkError;

        for (PetscInt bc = 0; bc < numberBC; bc++) {
            DMBoundaryConditionType type;
            const char* name;
            const char* labelName;
            PetscInt field;
            PetscInt numberIds;
            const PetscInt* ids;

            // Get the boundary
            PetscDSGetBoundary(flowProblem, bc, &type, &name, &labelName, &field, NULL, NULL, NULL, NULL, &numberIds, &ids, NULL) >> checkError;

            // If this is for euler and DM_BC_NATURAL_RIEMANN add it to the aux
            if (type == DM_BC_NATURAL_RIEMANN && field == 0) {
                for (PetscInt af = 0; af < numberAuxFields; af++) {
                    PetscDSAddBoundary(auxProblem, type, name, labelName, af, 0, NULL, NULL, NULL, numberIds, ids, NULL) >> checkError;
                }
            }
        }
    }
}
void ablate::flow::FVFlow::RegisterRHSFunction(FVMRHSFluxFunction function, void* context, std::string field, std::vector<std::string> inputFields, std::vector<std::string> auxFields) {
    // map the field, inputFields, and auxFields to locations
    auto fieldId = this->GetFieldId(field);
    if (!fieldId) {
        throw std::invalid_argument("Cannot locate flow field " + field);
    }

    // Create the FVMRHS Function
    FVMRHSFluxFunctionDescription functionDescription{.function = function,
                                                      .context = context,
                                                      .field = fieldId.value(),
                                                      .inputFields = {-1, -1, -1, -1}, /**default to empty.  Right now it is hard coded to be a 4 length array.  This should be relaxed**/
                                                      .numberInputFields = (PetscInt)inputFields.size(),
                                                      .auxFields = {-1, -1, -1, -1}, /**default to empty**/
                                                      .numberAuxFields = (PetscInt)auxFields.size()};

    if (inputFields.size() > MAX_FVM_RHS_FUNCTION_FIELDS || auxFields.size() > MAX_FVM_RHS_FUNCTION_FIELDS) {
        std::runtime_error("Cannot register more than " + std::to_string(MAX_FVM_RHS_FUNCTION_FIELDS) + " fields in RegisterRHSFunction.");
    }

    for (int i = 0; i < inputFields.size(); i++) {
        auto fieldId = this->GetFieldId(inputFields[i]);
        if (!fieldId) {
            throw std::invalid_argument("Cannot locate flow field " + inputFields[i]);
        }
        functionDescription.inputFields[i] = fieldId.value();
    }

    for (int i = 0; i < auxFields.size(); i++) {
        auto fieldId = this->GetAuxFieldId(auxFields[i]);
        if (!fieldId) {
            throw std::invalid_argument("Cannot locate flow field " + auxFields[i]);
        }
        functionDescription.auxFields[i] = fieldId.value();
    }

    rhsFluxFunctionDescriptions.push_back(functionDescription);
}


void ablate::flow::FVFlow::RegisterRHSFunction(FVMRHSPointFunction function, void* context, std::vector<std::string> fields, std::vector<std::string> inputFields, std::vector<std::string> auxFields) {
    // Create the FVMRHS Function
    FVMRHSPointFunctionDescription functionDescription{.function = function,
        .context = context,
        .fields = {-1, -1, -1, -1}, /**default to empty.  Right now it is hard coded to be a 4 length array.  This should be relaxed**/
        .numberFields = (PetscInt)fields.size(),
        .inputFields = {-1, -1, -1, -1}, /**default to empty.**/
        .numberInputFields = (PetscInt)inputFields.size(),
        .auxFields = {-1, -1, -1, -1}, /**default to empty**/
        .numberAuxFields = (PetscInt)auxFields.size()};

    if (fields.size() > MAX_FVM_RHS_FUNCTION_FIELDS || inputFields.size() > MAX_FVM_RHS_FUNCTION_FIELDS || auxFields.size() > MAX_FVM_RHS_FUNCTION_FIELDS) {
        std::runtime_error("Cannot register more than " + std::to_string(MAX_FVM_RHS_FUNCTION_FIELDS) + " fields in RegisterRHSFunction.");
    }

    for (int i = 0; i < fields.size(); i++) {
        auto fieldId = this->GetFieldId(fields[i]);
        if (!fieldId) {
            throw std::invalid_argument("Cannot locate flow field " + inputFields[i]);
        }
        functionDescription.fields[i] = fieldId.value();
    }

    for (int i = 0; i < inputFields.size(); i++) {
        auto fieldId = this->GetFieldId(inputFields[i]);
        if (!fieldId) {
            throw std::invalid_argument("Cannot locate flow field " + inputFields[i]);
        }
        functionDescription.inputFields[i] = fieldId.value();
    }

    for (int i = 0; i < auxFields.size(); i++) {
        auto fieldId = this->GetAuxFieldId(auxFields[i]);
        if (!fieldId) {
            throw std::invalid_argument("Cannot locate flow field " + auxFields[i]);
        }
        functionDescription.auxFields[i] = fieldId.value();
    }

    rhsPointFunctionDescriptions.push_back(functionDescription);
}

void ablate::flow::FVFlow::RegisterRHSFunction(RHSArbitraryFunction function, void* context){
    rhsArbitraryFunctions.push_back(std::make_pair(function, context));
}

void ablate::flow::FVFlow::RegisterAuxFieldUpdate(FVAuxFieldUpdateFunction function, void* context, std::string auxField) {
    // find the field location
    auto auxFieldLocation = this->GetAuxFieldId(auxField);

    if (!auxFieldLocation) {
        throw std::invalid_argument("Cannot locate aux flow field " + auxField);
    }

    // Make sure the items are sized correct
    auxFieldUpdateFunctions.resize(this->auxFieldDescriptors.size());
    auxFieldUpdateContexts.resize(this->auxFieldDescriptors.size());

    auxFieldUpdateFunctions[auxFieldLocation.value()] = function;
    auxFieldUpdateContexts[auxFieldLocation.value()] = context;
}
