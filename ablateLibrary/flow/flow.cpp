#include "flow.hpp"
#include "../utilities/petscError.hpp"
#include "../utilities/petscOptions.hpp"

ablate::flow::Flow::Flow(std::shared_ptr<mesh::Mesh> mesh, std::string name, std::map<std::string, std::string> arguments, std::vector<std::shared_ptr<FlowFieldSolution>> initialization,
                         std::vector<std::shared_ptr<BoundaryCondition>> boundaryConditions, std::vector<std::shared_ptr<FlowFieldSolution>> auxiliaryFields)
    : mesh(mesh), name(name), initialization(initialization), boundaryConditions(boundaryConditions), auxiliaryFields(auxiliaryFields) {
    // append any prefix values
    utilities::PetscOptions::Set(name, arguments);

    // setup the flow data
    FlowCreate(&flowData) >> checkError;
}

ablate::flow::Flow::~Flow() { FlowDestroy(&flowData) >> checkError; }

std::optional<int> ablate::flow::Flow::GetFieldId(const std::string& fieldName) {
    for (auto f = 0; f < flowData->numberFlowFields; f++) {
        if (flowData->flowFieldDescriptors[f].fieldName == fieldName) {
            return f;
        }
    }
    return {};
}

std::optional<int> ablate::flow::Flow::GetAuxFieldId(const std::string& fieldName) {
    for (auto f = 0; f < flowData->numberAuxFields; f++) {
        if (flowData->auxFieldDescriptors[f].fieldName == fieldName) {
            return f;
        }
    }
    return {};
}

void ablate::flow::Flow::CompleteInitialization() {
    // Apply any boundary conditions
    PetscDS prob;
    DMGetDS(mesh->GetDomain(), &prob) >> checkError;

    // add each boundary condition
    for (auto boundary : boundaryConditions) {
        PetscInt id = boundary->GetLabelId();
        auto fieldId = GetFieldId(boundary->GetFieldName());
        if (!fieldId) {
            throw std::invalid_argument("unknown field for boundary: " + boundary->GetFieldName());
        }

        // Get the current field type
        DMBoundaryConditionType boundaryConditionType;
        switch (flowData->flowFieldDescriptors[fieldId.value()].fieldType) {
            case FE:
                boundaryConditionType = DM_BC_ESSENTIAL;
                break;
            case FV:
                boundaryConditionType = DM_BC_NATURAL_RIEMANN;
                break;
            default:
                throw std::invalid_argument("unknown field type " + std::to_string(flowData->flowFieldDescriptors[fieldId.value()].fieldType));
        }

        PetscDSAddBoundary(prob,
                           boundaryConditionType,
                           boundary->GetBoundaryName().c_str(),
                           boundary->GetLabelName().c_str(),
                           fieldId.value(),
                           0,
                           NULL,
                           (void (*)(void))boundary->GetBoundaryFunction(),
                           (void (*)(void))boundary->GetBoundaryTimeDerivativeFunction(),
                           1,
                           &id,
                           boundary->GetContext()) >>
            checkError;
    }

    // Set the exact solution
    for (auto exact : initialization) {
        auto fieldId = GetFieldId(exact->GetName());
        if (!fieldId) {
            throw std::invalid_argument("unknown field for initialization: " + exact->GetName());
        }

        PetscDSSetExactSolution(prob, fieldId.value(), exact->GetSolutionField().GetPetscFunction(), exact->GetSolutionField().GetContext()) >> checkError;
        PetscDSSetExactSolutionTimeDerivative(prob, fieldId.value(), exact->GetTimeDerivative().GetPetscFunction(), exact->GetTimeDerivative().GetContext()) >> checkError;
    }

    // check to see if
    if (flowData->numberAuxFields > 0 && !auxiliaryFields.empty()) {
        PetscInt numberAuxFields;
        DMGetNumFields(flowData->auxDm, &numberAuxFields) >> checkError;

        // size up the update and context functions
        auxiliaryFieldContexts.resize(numberAuxFields, NULL);
        auxiliaryFieldFunctions.resize(numberAuxFields, NULL);

        // for each given aux field
        for (auto auxFieldDescription : auxiliaryFields) {
            auto fieldId = GetAuxFieldId(auxFieldDescription->GetName());
            if (!fieldId) {
                throw std::invalid_argument("unknown field for aux field: " + auxFieldDescription->GetName());
            }

            auxiliaryFieldContexts[fieldId.value()] = auxFieldDescription->GetSolutionField().GetContext();
            auxiliaryFieldFunctions[fieldId.value()] = auxFieldDescription->GetSolutionField().GetPetscFunction();
        }

        FlowRegisterPreStep(flowData, UpdateAuxiliaryFields, this);
    }
}

PetscErrorCode ablate::flow::Flow::UpdateAuxiliaryFields(TS ts, void* ctx) {
    PetscFunctionBegin;
    PetscErrorCode ierr;
    auto flow = (Flow*)ctx;

    // get the time at the end of the time step
    PetscReal time = 0;
    PetscReal dt = 0;

    ierr = TSGetTime(ts, &time);
    CHKERRQ(ierr);
    if (time > 0) {
        ierr = TSGetTimeStep(ts, &dt);
        CHKERRQ(ierr);
    }
    // Update the source terms
    ierr = DMProjectFunctionLocal(flow->GetFlowData()->auxDm, time + dt, &(flow->auxiliaryFieldFunctions[0]), &(flow->auxiliaryFieldContexts[0]), INSERT_ALL_VALUES, flow->GetFlowData()->auxField);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

void ablate::flow::Flow::SetupSolve(TS& timeStepper) {
    // Initialize the flow field
    DMComputeExactSolution(mesh->GetDomain(), 0, flowData->flowField, NULL) >> checkError;

    // Initialize the aux variables
    if (flowData->numberAuxFields > 0) {
        UpdateAuxiliaryFields(timeStepper, this) >> checkError;
    }

    // Re Name the flow field
    PetscObjectSetName((PetscObject)(flowData->flowField), this->name.c_str()) >> checkError;
    VecSetOptionsPrefix(flowData->flowField, "num_sol_") >> checkError;
}
