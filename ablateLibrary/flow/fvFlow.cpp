#include "fvFlow.hpp"
#include <flow/processes/flowProcess.hpp>
#include <utilities/mpiError.hpp>
#include <utilities/petscError.hpp>

ablate::flow::FVFlow::FVFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<parameters::Parameters> parameters, std::vector<FlowFieldDescriptor> fieldDescriptors,
                             std::vector<std::shared_ptr<processes::FlowProcess>> flowProcessesIn, std::shared_ptr<parameters::Parameters> options,
                             std::vector<std::shared_ptr<mathFunctions::FieldSolution>> initialization, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                             std::vector<std::shared_ptr<mathFunctions::FieldSolution>> auxiliaryFields, std::vector<std::shared_ptr<mathFunctions::FieldSolution>> exactSolution)
    : Flow(name, mesh, parameters, options, initialization, boundaryConditions, auxiliaryFields, exactSolution), flowProcesses(flowProcessesIn) {
    // make sure that the dm works with fv
    const PetscInt ghostCellDepth = 1;
    DM& dm = this->dm->GetDomain();
    {  // Make sure that the flow is setup distributed
        DM dmDist;
        DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE) >> checkError;
        DMPlexDistribute(dm, ghostCellDepth, NULL, &dmDist) >> checkError;
        if (dmDist) {
            DMDestroy(&dm) >> checkError;
            dm = dmDist;
        }
    }

    // create any ghost cells that are needed
    {
        DM gdm;
        DMPlexConstructGhostCells(dm, NULL, NULL, &gdm) >> checkError;
        DMDestroy(&dm) >> checkError;
        dm = gdm;
    }

    // Copy over the application context if needed
    DMSetApplicationContext(dm, this) >> checkError;

    // initialize each field
    for (const auto& field : fieldDescriptors) {
        if (field.components != 0) {
            RegisterField(field);
        }
    }
    FinalizeRegisterFields();

    // march over process and link to the flow
    for (const auto& process : flowProcesses) {
        process->Initialize(*this);
    }

    // Start problem setup
    PetscDS prob;
    DMGetDS(dm, &prob) >> checkError;

    // Set the flux calculator solver for each component
    PetscDSSetFromOptions(prob) >> checkError;
}

ablate::flow::FVFlow::FVFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<parameters::Parameters> parameters, std::vector<std::shared_ptr<FlowFieldDescriptor>> fieldDescriptors,
                             std::vector<std::shared_ptr<processes::FlowProcess>> flowProcessesIn, std::shared_ptr<parameters::Parameters> options,
                             std::vector<std::shared_ptr<mathFunctions::FieldSolution>> initialization, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                             std::vector<std::shared_ptr<mathFunctions::FieldSolution>> auxiliaryFields, std::vector<std::shared_ptr<mathFunctions::FieldSolution>> exactSolution)
    : ablate::flow::FVFlow::FVFlow(
          name, mesh, parameters,
          [](auto fieldDescriptorsPtrs) {
              auto vec = std::vector<FlowFieldDescriptor>{};
              for (auto ptr : fieldDescriptorsPtrs) {
                  vec.push_back(*ptr);
              }
              return vec;
          }(fieldDescriptors),
          flowProcessesIn, options, initialization, boundaryConditions, auxiliaryFields, exactSolution) {}

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
    ierr = ABLATE_DMPlexComputeRHSFunctionFVM(&flow->rhsFluxFunctionDescriptions[0],
                                              flow->rhsFluxFunctionDescriptions.size(),
                                              &flow->rhsPointFunctionDescriptions[0],
                                              flow->rhsPointFunctionDescriptions.size(),
                                              dm,
                                              time,
                                              locXVec,
                                              globFVec);
    CHKERRQ(ierr);

    // iterate over any arbitrary RHS functions
    for (const auto& rhsFunction : flow->rhsArbitraryFunctions) {
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

    if (!timeStepFunctions.empty()) {
        preStepFunctions.push_back(ComputeTimeStep);
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

void ablate::flow::FVFlow::RegisterRHSFunction(RHSArbitraryFunction function, void* context) { rhsArbitraryFunctions.push_back(std::make_pair(function, context)); }

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

void ablate::flow::FVFlow::ComputeTimeStep(TS ts, ablate::flow::Flow& flow) {
    // Get the dm and current solution vector
    DM dm;
    TSGetDM(ts, &dm) >> checkError;

    // Get the flow param
    ablate::flow::FVFlow& flowFV = dynamic_cast<ablate::flow::FVFlow&>(flow);

    // march over each calculator
    PetscReal dtMin = 1000.0;
    for (const auto& dtFunction : flowFV.timeStepFunctions) {
        dtMin = PetscMin(dtMin, dtFunction.first(ts, flow, dtFunction.second));
    }

    // take the min across all ranks
    PetscInt rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)ts), &rank);

    PetscReal dtMinGlobal;
    MPI_Allreduce(&dtMin, &dtMinGlobal, 1, MPIU_REAL, MPI_MIN, PetscObjectComm((PetscObject)ts)) >> checkMpiError;

    TSSetTimeStep(ts, dtMinGlobal) >> checkError;
    if (PetscIsNanReal(dtMinGlobal)) {
        throw std::runtime_error("Invalid timestep selected for flow");
    }
}
void ablate::flow::FVFlow::RegisterComputeTimeStepFunction(ComputeTimeStepFunction function, void* ctx) { timeStepFunctions.push_back(std::make_pair(function, ctx)); }

#include "parser/registrar.hpp"
REGISTER(ablate::flow::Flow, ablate::flow::FVFlow, "finite volume flow", ARG(std::string, "name", "the name of the flow field"), ARG(ablate::mesh::Mesh, "mesh", "the  mesh and discretization"),
         OPT(ablate::parameters::Parameters, "parameters", "the parameters used by the flow"), ARG(std::vector<ablate::flow::FlowFieldDescriptor>, "fields", "field descriptions"),
         ARG(std::vector<ablate::flow::processes::FlowProcess>, "processes", "the processes used to describe the flow"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"), OPT(std::vector<mathFunctions::FieldSolution>, "initialization", "the flow field initialization"),
         OPT(std::vector<flow::boundaryConditions::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"),
         OPT(std::vector<mathFunctions::FieldSolution>, "auxiliaryFields", "the aux flow field initialization"),
         OPT(std::vector<mathFunctions::FieldSolution>, "exactSolution", "optional exact solutions that can be used for error calculations"));