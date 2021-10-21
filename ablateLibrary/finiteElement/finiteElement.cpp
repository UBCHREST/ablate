
#include "finiteElement.hpp"
#include <utilities/mpiError.hpp>
#include <utilities/petscError.hpp>

ablate::finiteElement::FiniteElement::FiniteElement(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options,
                                                    std::vector<ablate::domain::FieldDescriptor> fieldDescriptors, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization,
                                                    std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                    std::vector<std::shared_ptr<mathFunctions::FieldFunction>> auxiliaryFields,
                                                    std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolution)
    : Solver(solverId, region, options),
      fieldDescriptors(fieldDescriptors),
      initialization(initialization),
      boundaryConditions(boundaryConditions),
      auxiliaryFieldsUpdaters(auxiliaryFields),
      exactSolutions(exactSolution) {}

void ablate::finiteElement::FiniteElement::Register(std::shared_ptr<ablate::domain::SubDomain> subDomain) {
    Solver::Register(subDomain);
    Solver::DecompressFieldFieldDescriptor(fieldDescriptors);

    // initialize each field
    for (const auto& field : fieldDescriptors) {
        if (!field.components.empty()) {
            RegisterFiniteElementField(field);
        }
    }
}

void ablate::finiteElement::FiniteElement::Setup() {
    DM cdm = subDomain->GetDM();
    while (cdm) {
        DMCopyDisc(subDomain->GetDM(), cdm) >> checkError;
        DMGetCoarseDM(cdm, &cdm) >> checkError;
    }

    // Register the aux fields updater if specified
    if (!auxiliaryFieldsUpdaters.empty()) {
        RegisterPreStep([&](TS ts, Solver&) { UpdateAuxFields(ts, *this); });
    }

    // Apply any boundary conditions
    PetscDS prob;
    DMGetDS(subDomain->GetDM(), &prob) >> checkError;

    // add each boundary condition
    for (auto boundary : boundaryConditions) {
        const auto& fieldId = subDomain->GetField(boundary->GetFieldName());

        // Setup the boundary condition
        boundary->SetupBoundary(subDomain->GetDM(), prob, fieldId.id);
    }
}

void ablate::finiteElement::FiniteElement::Initialize() {
    // Apply any boundary conditions
    PetscDS prob;
    DMGetDS(subDomain->GetDM(), &prob) >> checkError;

    // Initialize the flow field if provided
    if (!initialization.empty()) {
        PetscInt numberFields;
        DMGetNumFields(subDomain->GetDM(), &numberFields) >> checkError;

        // size up the update and context functions
        std::vector<mathFunctions::PetscFunction> fieldFunctions(numberFields, NULL);
        std::vector<void*> fieldContexts(numberFields, NULL);

        for (auto fieldInitialization : initialization) {
            auto fieldId = subDomain->GetField(fieldInitialization->GetName());

            fieldContexts[fieldId.id] = fieldInitialization->GetSolutionField().GetContext();
            fieldFunctions[fieldId.id] = fieldInitialization->GetSolutionField().GetPetscFunction();
        }

        DMProjectFunction(subDomain->GetDM(), 0.0, &fieldFunctions[0], &fieldContexts[0], INSERT_VALUES, subDomain->GetSolutionVector()) >> checkError;
        this->CompleteFlowInitialization(subDomain->GetDM(), subDomain->GetSolutionVector());
    }

    // if an exact solution has been provided register it
    for (const auto& exactSolution : exactSolutions) {
        auto fieldId = subDomain->GetField(exactSolution->GetName());

        // Get the current field type
        if (exactSolution->HasSolutionField()) {
            PetscDSSetExactSolution(prob, fieldId.id, exactSolution->GetSolutionField().GetPetscFunction(), exactSolution->GetSolutionField().GetContext()) >> checkError;
        }
        if (exactSolution->HasTimeDerivative()) {
            PetscDSSetExactSolutionTimeDerivative(prob, fieldId.id, exactSolution->GetTimeDerivative().GetPetscFunction(), exactSolution->GetTimeDerivative().GetContext()) >> checkError;
        }
    }

    DMTSSetBoundaryLocal(subDomain->GetDM(), DMPlexTSComputeBoundary, NULL) >> checkError;
    DMTSSetIFunctionLocal(subDomain->GetDM(), DMPlexTSComputeIFunctionFEM, NULL) >> checkError;
    DMTSSetIJacobianLocal(subDomain->GetDM(), DMPlexTSComputeIJacobianFEM, NULL) >> checkError;

    // copy over any boundary information from the dm, to the aux dm and set the sideset
    if (subDomain->GetAuxDM()) {
        PetscDS flowProblem;
        DMGetDS(subDomain->GetDM(), &flowProblem) >> checkError;
        PetscDS auxProblem;
        DMGetDS(subDomain->GetAuxDM(), &auxProblem) >> checkError;

        // Get the number of boundary conditions and other info
        PetscInt numberBC;
        PetscDSGetNumBoundary(flowProblem, &numberBC) >> checkError;
        PetscInt numberAuxFields;
        PetscDSGetNumFields(auxProblem, &numberAuxFields) >> checkError;

        for (PetscInt bc = 0; bc < numberBC; bc++) {
            DMBoundaryConditionType type;
            const char* name;
            DMLabel label;
            PetscInt field;
            PetscInt numberIds;
            const PetscInt* ids;

            // Get the boundary
            PetscDSGetBoundary(flowProblem, bc, NULL, &type, &name, &label, &numberIds, &ids, &field, NULL, NULL, NULL, NULL, NULL) >> checkError;

            // If this is for euler and DM_BC_NATURAL_RIEMANN add it to the aux
            if (type == DM_BC_NATURAL_RIEMANN && field == 0) {
                for (PetscInt af = 0; af < numberAuxFields; af++) {
                    PetscDSAddBoundary(auxProblem, type, name, label, numberIds, ids, af, 0, NULL, NULL, NULL, NULL, NULL) >> checkError;
                }
            }
        }
    }
}

void ablate::finiteElement::FiniteElement::UpdateAuxFields(TS ts, ablate::finiteElement::FiniteElement& fe) {
    PetscInt numberAuxFields;
    DMGetNumFields(fe.subDomain->GetAuxDM(), &numberAuxFields) >> checkError;

    // size up the update and context functions
    std::vector<mathFunctions::PetscFunction> auxiliaryFieldFunctions(numberAuxFields, NULL);
    std::vector<void*> auxiliaryFieldContexts(numberAuxFields, NULL);

    // for each given aux field
    for (auto auxFieldDescription : fe.auxiliaryFieldsUpdaters) {
        auto fieldId = fe.subDomain->GetField(auxFieldDescription->GetName());
        auxiliaryFieldContexts[fieldId.id] = auxFieldDescription->GetSolutionField().GetContext();
        auxiliaryFieldFunctions[fieldId.id] = auxFieldDescription->GetSolutionField().GetPetscFunction();
    }

    // get the time at the end of the time step
    PetscReal time = 0;
    PetscReal dt = 0;
    TSGetTime(ts, &time) >> checkError;
    TSGetTimeStep(ts, &dt) >> checkError;

    // Update the source terms
    DMProjectFunctionLocal(fe.subDomain->GetAuxDM(), time + dt, &auxiliaryFieldFunctions[0], &auxiliaryFieldContexts[0], INSERT_ALL_VALUES, fe.subDomain->GetAuxVector()) >> checkError;
}

void ablate::finiteElement::FiniteElement::RegisterFiniteElementField(const ablate::domain::FieldDescriptor& fieldDescriptor) {
    // determine if it a simplex element and the number of dimensions
    DMPolytopeType ct;
    PetscInt cStart;
    DMPlexGetHeightStratum(subDomain->GetDM(), 0, &cStart, NULL) >> checkError;
    DMPlexGetCellType(subDomain->GetDM(), cStart, &ct) >> checkError;
    PetscInt simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1 ? PETSC_TRUE : PETSC_FALSE;
    PetscInt simplexGlobal;

    // Assume true if any rank says true
    MPI_Allreduce(&simplex, &simplexGlobal, 1, MPIU_INT, MPI_MAX, subDomain->GetComm()) >> checkMpiError;
    // create a petsc fe
    PetscFE petscFE;
    PetscFECreateDefault(PetscObjectComm((PetscObject)subDomain->GetDM()),
                         subDomain->GetDimensions(),
                         fieldDescriptor.components.size(),
                         simplexGlobal ? PETSC_TRUE : PETSC_FALSE,
                         fieldDescriptor.prefix.c_str(),
                         PETSC_DEFAULT,
                         &petscFE) >>
        checkError;
    PetscObjectSetName((PetscObject)petscFE, fieldDescriptor.name.c_str()) >> checkError;
    PetscObjectSetOptions((PetscObject)petscFE, petscOptions) >> checkError;

    // If this is not the first field, copy the quadrature locations
    if (subDomain->GetNumberFields() > 0) {
        PetscFE referencePetscFE;
        DMGetField(subDomain->GetDM(), 0, NULL, (PetscObject*)&referencePetscFE) >> checkError;
        PetscFECopyQuadrature(referencePetscFE, petscFE) >> checkError;
    }

    // Register the field with the subDomain
    subDomain->RegisterField(fieldDescriptor, (PetscObject)petscFE);

    PetscFEDestroy(&petscFE) >> checkError;
}

void ablate::finiteElement::FiniteElement::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) const {
    Solver::Save(viewer, sequenceNumber, time);

    if (!exactSolutions.empty()) {
        Vec exactVec;
        DMGetGlobalVector(subDomain->GetDM(), &exactVec) >> checkError;

        // Get the number of fields
        PetscDS ds;
        DMGetDS(subDomain->GetDM(), &ds) >> checkError;
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

        DMProjectFunction(subDomain->GetDM(), time, &exactFuncs[0], &exactCtxs[0], INSERT_ALL_VALUES, exactVec) >> checkError;

        PetscObjectSetName((PetscObject)exactVec, "exact") >> checkError;
        VecView(exactVec, viewer) >> checkError;
        DMRestoreGlobalVector(subDomain->GetDM(), &exactVec) >> checkError;
    }
}