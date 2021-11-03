#include "finiteVolume.hpp"
#include <petsc/private/dmpleximpl.h>
#include <utilities/mpiError.hpp>
#include <utilities/petscError.hpp>
#include "processes/process.hpp"

ablate::finiteVolume::FiniteVolume::FiniteVolume(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options,
                                                 std::vector<ablate::domain::FieldDescriptor> fieldDescriptors, std::vector<std::shared_ptr<processes::Process>> processes,
                                                 std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization,
                                                 std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                 std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolution)
    : Solver(std::move(solverId), std::move(region), std::move(options)),
      processes(std::move(processes)),
      fieldDescriptors(std::move(fieldDescriptors)),
      initialization(std::move(initialization)),
      boundaryConditions(std::move(boundaryConditions)),
      exactSolutions(std::move(exactSolution)) {}

ablate::finiteVolume::FiniteVolume::FiniteVolume(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options,
                                                 std::vector<std::shared_ptr<domain::FieldDescriptor>> fieldDescriptors, std::vector<std::shared_ptr<processes::Process>> processes,
                                                 std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization,
                                                 std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                 std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolution)
    : ablate::finiteVolume::FiniteVolume::FiniteVolume(
          std::move(solverId), std::move(region), std::move(options),
          [](auto fieldDescriptorsPtrs) {
              auto vec = std::vector<domain::FieldDescriptor>{};
              for (const auto& ptr : fieldDescriptorsPtrs) {
                  vec.push_back(*ptr);
              }
              return vec;
          }(std::move(fieldDescriptors)),
          std::move(processes), std::move(initialization), std::move(boundaryConditions), std::move(exactSolution)) {}
ablate::finiteVolume::FiniteVolume::~FiniteVolume() {}

void ablate::finiteVolume::FiniteVolume::Register(std::shared_ptr<ablate::domain::SubDomain> subDomain) {
    Solver::Register(subDomain);
    Solver::DecompressFieldFieldDescriptor(fieldDescriptors);

    // initialize each field
    for (auto& field : fieldDescriptors) {
        if (!field.components.empty()) {
            // check the field adjacency
            if (field.adjacency == domain::FieldAdjacency::DEFAULT) {
                field.adjacency = domain::FieldAdjacency::FVM;
            }

            RegisterFiniteVolumeField(field);
        }
    }
}

void ablate::finiteVolume::FiniteVolume::Setup() {
    // march over process and link to the flow
    for (const auto& process : processes) {
        process->Initialize(*this);
    }

    // Set the flux calculator solver for each component
    PetscDSSetFromOptions(subDomain->GetDiscreteSystem()) >> checkError;
}

void ablate::finiteVolume::FiniteVolume::Initialize() {
    // add each boundary condition
    for (auto boundary : boundaryConditions) {
        const auto& fieldId = subDomain->GetSolutionField(boundary->GetFieldName());

        // Setup the boundary condition
        boundary->SetupBoundary(subDomain->GetDM(), subDomain->GetDiscreteSystem(), fieldId.id);
    }

    // Initialize the flow field if provided
    subDomain->ProjectFieldFunctions(initialization, subDomain->GetSolutionVector());

    // if an exact solution has been provided register it
    for (const auto& exactSolution : exactSolutions) {
        auto fieldId = subDomain->GetField(exactSolution->GetName());

        // Get the current field type
        if (exactSolution->HasSolutionField()) {
            PetscDSSetExactSolution(subDomain->GetDiscreteSystem(), fieldId.id, exactSolution->GetSolutionField().GetPetscFunction(), exactSolution->GetSolutionField().GetContext()) >> checkError;
        }
        if (exactSolution->HasTimeDerivative()) {
            PetscDSSetExactSolutionTimeDerivative(subDomain->GetDiscreteSystem(), fieldId.id, exactSolution->GetTimeDerivative().GetPetscFunction(), exactSolution->GetTimeDerivative().GetContext()) >>
                checkError;
        }
    }

    // copy over any boundary information from the dm, to the aux dm and set the sideset
    if (subDomain->GetAuxDM()) {
        PetscDS flowProblem = subDomain->GetDiscreteSystem();
        PetscDS auxProblem = subDomain->GetAuxDiscreteSystem();

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
    if (!timeStepFunctions.empty()) {
        RegisterPreStep(ComputeTimeStep);
    }
}

void ablate::finiteVolume::FiniteVolume::RegisterFiniteVolumeField(const ablate::domain::FieldDescriptor& fieldDescriptor) {
    PetscFV fvm;
    PetscFVCreate(PetscObjectComm((PetscObject)subDomain->GetDM()), &fvm) >> checkError;
    PetscObjectSetOptionsPrefix((PetscObject)fvm, fieldDescriptor.prefix.c_str()) >> checkError;
    PetscObjectSetName((PetscObject)fvm, fieldDescriptor.name.c_str()) >> checkError;
    PetscObjectSetOptions((PetscObject)fvm, petscOptions) >> checkError;

    PetscFVSetFromOptions(fvm) >> checkError;
    PetscFVSetNumComponents(fvm, fieldDescriptor.components.size()) >> checkError;
    PetscFVSetSpatialDimension(fvm, subDomain->GetDimensions()) >> checkError;

    // If there are any names provided, name each component in this field this is used by some of the output fields
    for (std::size_t c = 0; c < fieldDescriptor.components.size(); c++) {
        PetscFVSetComponentName(fvm, c, fieldDescriptor.components[c].c_str()) >> checkError;
    }

    // Register the field with the subDomain
    subDomain->RegisterField(fieldDescriptor, (PetscObject)fvm);

    PetscFVDestroy(&fvm) >> checkError;
}

PetscErrorCode ablate::finiteVolume::FiniteVolume::ComputeRHSFunction(PetscReal time, Vec locXVec, Vec locFVec) {
    PetscFunctionBeginUser;

    PetscErrorCode ierr;

    auto dm = subDomain->GetDM();
    auto ds = subDomain->GetDiscreteSystem();
    /* Handle non-essential (e.g. outflow) boundary values.  This should be done before the auxFields are updated so that boundary values can be updated */
    Vec facegeom, cellgeom;
    ierr = DMPlexGetGeometryFVM(dm, &facegeom, &cellgeom, NULL);
    CHKERRQ(ierr);
    ierr = ablate::solver::Solver::DMPlexInsertBoundaryValues_Plex(dm, ds, PETSC_FALSE, locXVec, time, facegeom, cellgeom, NULL);
    CHKERRQ(ierr);

    try {
        // update any aux fields, including ghost cells
        UpdateAuxFields(time, locXVec, subDomain->GetAuxVector());

        // Compute the RHS function
        //        ComputeFlux(time, locXVec, subDomain->GetAuxVector(), locFVec);
    } catch (std::exception& exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exception.what());
    }

    // compute the  flux across each face and point wise functions(note CompressibleFlowComputeEulerFlux has already been registered)
    ierr = ABLATE_DMPlexComputeRHSFunctionFVM(
        &rhsFluxFunctionDescriptions[0], rhsFluxFunctionDescriptions.size(), &rhsPointFunctionDescriptions[0], rhsPointFunctionDescriptions.size(), dm, time, locXVec, locFVec);
    CHKERRQ(ierr);

    // iterate over any arbitrary RHS functions
    for (const auto& rhsFunction : rhsArbitraryFunctions) {
        ierr = rhsFunction.first(dm, time, locXVec, locFVec, rhsFunction.second);
        CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}

void ablate::finiteVolume::FiniteVolume::RegisterRHSFunction(FVMRHSFluxFunction function, void* context, std::string field, std::vector<std::string> inputFields, std::vector<std::string> auxFields) {
    // map the field, inputFields, and auxFields to locations
    auto& fieldId = subDomain->GetField(field);

    // Create the FVMRHS Function
    FVMRHSFluxFunctionDescription functionDescription{.function = function,
                                                      .context = context,
                                                      .field = fieldId.id,
                                                      .inputFields = {-1, -1, -1, -1}, /**default to empty.  Right now it is hard coded to be a 4 length array.  This should be relaxed**/
                                                      .numberInputFields = (PetscInt)inputFields.size(),
                                                      .auxFields = {-1, -1, -1, -1}, /**default to empty**/
                                                      .numberAuxFields = (PetscInt)auxFields.size()};

    if (inputFields.size() > MAX_FVM_RHS_FUNCTION_FIELDS || auxFields.size() > MAX_FVM_RHS_FUNCTION_FIELDS) {
        std::runtime_error("Cannot register more than " + std::to_string(MAX_FVM_RHS_FUNCTION_FIELDS) + " fields in RegisterRHSFunction.");
    }

    for (std::size_t i = 0; i < inputFields.size(); i++) {
        auto& inputFieldId = subDomain->GetField(inputFields[i]);
        functionDescription.inputFields[i] = inputFieldId.id;
    }

    for (std::size_t i = 0; i < auxFields.size(); i++) {
        auto& auxFieldId = subDomain->GetField(auxFields[i]);
        functionDescription.auxFields[i] = auxFieldId.id;
    }

    rhsFluxFunctionDescriptions.push_back(functionDescription);
}

void ablate::finiteVolume::FiniteVolume::RegisterRHSFunction(FVMRHSPointFunction function, void* context, std::vector<std::string> fields, std::vector<std::string> inputFields,
                                                             std::vector<std::string> auxFields) {
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

    for (std::size_t i = 0; i < fields.size(); i++) {
        auto& fieldId = subDomain->GetField(fields[i]);
        functionDescription.fields[i] = fieldId.id;
    }

    for (std::size_t i = 0; i < inputFields.size(); i++) {
        auto& fieldId = subDomain->GetField(inputFields[i]);
        functionDescription.inputFields[i] = fieldId.id;
    }

    for (std::size_t i = 0; i < auxFields.size(); i++) {
        auto& fieldId = subDomain->GetField(auxFields[i]);
        functionDescription.auxFields[i] = fieldId.id;
    }

    rhsPointFunctionDescriptions.push_back(functionDescription);
}

void ablate::finiteVolume::FiniteVolume::RegisterRHSFunction(RHSArbitraryFunction function, void* context) { rhsArbitraryFunctions.push_back(std::make_pair(function, context)); }

void ablate::finiteVolume::FiniteVolume::RegisterAuxFieldUpdate(AuxFieldUpdateFunction function, void* context, std::string auxField, std::vector<std::string> inputFields) {
    // find the field location
    auto& auxFieldLocation = subDomain->GetField(auxField);

    AuxFieldUpdateFunctionDescription functionDescription{.function = function, .context = context, .inputFields = {}, .auxField = auxFieldLocation.id};

    for (std::size_t i = 0; i < inputFields.size(); i++) {
        auto fieldId = subDomain->GetField(inputFields[i]);
        functionDescription.inputFields.push_back(fieldId.id);
    }

    auxFieldUpdateFunctionDescriptions.push_back(functionDescription);
}

void ablate::finiteVolume::FiniteVolume::ComputeTimeStep(TS ts, ablate::solver::Solver& solver) {
    auto& flowFV = static_cast<ablate::finiteVolume::FiniteVolume&>(solver);
    // Get the dm and current solution vector
    DM dm;
    TSGetDM(ts, &dm) >> checkError;
    PetscInt timeStep;
    TSGetStepNumber(ts, &timeStep) >> checkError;
    PetscReal currentDt;
    TSGetTimeStep(ts, &currentDt) >> checkError;

    // march over each calculator
    PetscReal dtMin = 1000.0;
    for (const auto& dtFunction : flowFV.timeStepFunctions) {
        dtMin = PetscMin(dtMin, dtFunction.first(ts, flowFV, dtFunction.second));
    }

    // take the min across all ranks
    PetscMPIInt rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)ts), &rank);

    PetscReal dtMinGlobal;
    MPI_Allreduce(&dtMin, &dtMinGlobal, 1, MPIU_REAL, MPI_MIN, PetscObjectComm((PetscObject)ts)) >> checkMpiError;

    // don't override the first time step if bigger
    if (timeStep > 0 || dtMinGlobal < currentDt) {
        TSSetTimeStep(ts, dtMinGlobal) >> checkError;
        if (PetscIsNanReal(dtMinGlobal)) {
            throw std::runtime_error("Invalid timestep selected for flow");
        }
    }
}

void ablate::finiteVolume::FiniteVolume::RegisterComputeTimeStepFunction(ComputeTimeStepFunction function, void* ctx) { timeStepFunctions.push_back(std::make_pair(function, ctx)); }
void ablate::finiteVolume::FiniteVolume::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) const {
    Solver::Save(viewer, sequenceNumber, time);

    if (!exactSolutions.empty()) {
        Vec exactVec;
        DMGetGlobalVector(subDomain->GetSubDM(), &exactVec) >> checkError;

        subDomain->ProjectFieldFunctionsToSubDM(exactSolutions, exactVec, time);

        PetscObjectSetName((PetscObject)exactVec, "exact") >> checkError;
        VecView(exactVec, viewer) >> checkError;
        DMRestoreGlobalVector(subDomain->GetSubDM(), &exactVec) >> checkError;
    }
}

void ablate::finiteVolume::FiniteVolume::UpdateAuxFields(PetscReal time, Vec locXVec, Vec locAuxField) {
    DM plex;
    DM auxDM = GetSubDomain().GetAuxDM();
    // Convert to a dmplex
    DMConvert(GetSubDomain().GetDM(), DMPLEX, &plex) >> checkError;

    // Get the valid cell range over this region
    IS cellIS;
    PetscInt cStart, cEnd;
    const PetscInt* cells;
    GetCellRange(cellIS, cStart, cEnd, cells);

    // Extract the cell geometry, and the dm that holds the information
    Vec cellGeomVec;
    DM dmCell;
    const PetscScalar* cellGeomArray;
    DMPlexGetGeometryFVM(plex, NULL, &cellGeomVec, NULL) >> checkError;
    VecGetDM(cellGeomVec, &dmCell) >> checkError;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;

    // extract the low flow and aux fields
    const PetscScalar* locFlowFieldArray;
    VecGetArrayRead(locXVec, &locFlowFieldArray) >> checkError;

    PetscScalar* localAuxFlowFieldArray;
    VecGetArray(locAuxField, &localAuxFlowFieldArray) >> checkError;

    // Get the cell dim
    PetscInt dim = subDomain->GetDimensions();

    // determine the number of fields and the totDim
    PetscInt nf;
    PetscDSGetNumFields(subDomain->GetDiscreteSystem(), &nf) >> checkError;

    // Create the required offset arrays. These are sized for the max possible value
    PetscInt* uOff = NULL;
    PetscCalloc1(nf, &uOff) >> checkError;

    PetscInt* uOffTotal;
    PetscDSGetComponentOffsets(subDomain->GetDiscreteSystem(), &uOffTotal) >> checkError;

    // March over each cell volume
    for (PetscInt c = cStart; c < cEnd; ++c) {
        PetscFVCellGeom* cellGeom;
        const PetscReal* fieldValues;
        PetscReal* auxValues;

        // Get the cell location
        const PetscInt cell = cells ? cells[c] : c;

        DMPlexPointLocalRead(dmCell, cell, cellGeomArray, &cellGeom) >> checkError;
        DMPlexPointLocalRead(plex, cell, locFlowFieldArray, &fieldValues) >> checkError;

        // for each function description
        for (const auto& updateFunction : auxFieldUpdateFunctionDescriptions) {
            // get the uOff for the req fields
            for (std::size_t rf = 0; rf < updateFunction.inputFields.size(); rf++) {
                uOff[rf] = uOffTotal[updateFunction.inputFields[rf]];
            }

            // grab the local aux field
            DMPlexPointLocalFieldRef(auxDM, cell, updateFunction.auxField, localAuxFlowFieldArray, &auxValues) >> checkError;

            // If an update function was passed
            updateFunction.function(time, dim, cellGeom, uOff, fieldValues, auxValues, updateFunction.context) >> checkError;
        }
    }

    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    VecRestoreArrayRead(locXVec, &locFlowFieldArray) >> checkError;
    VecRestoreArray(locAuxField, &localAuxFlowFieldArray) >> checkError;

    RestoreRange(cellIS, cStart, cEnd, cells);

    DMDestroy(&plex) >> checkError;
    PetscFree(uOff) >> checkError;
}
void ablate::finiteVolume::FiniteVolume::ComputeFlux(PetscReal time, Vec locXVec, Vec locAuxField, Vec locF) {
    // Get the valid cell range over this region
    IS faceIS;
    PetscInt fStart, fEnd;
    const PetscInt* faces;
    GetFaceRange(faceIS, fStart, fEnd, faces);

    for (PetscInt f = fStart; f < fEnd; ++f) {
        const PetscInt face = faces ? faces[f] : f;

        std::cout << "face (" << f << "): " << face << std::endl;
    }

    RestoreRange(faceIS, fStart, fEnd, faces);
}

void ablate::finiteVolume::FiniteVolume::GetCellRange(IS& cellIS, PetscInt& cStart, PetscInt& cEnd, const PetscInt*& cells) {
    // Start out getting all of the cells
    PetscInt depth;
    DMPlexGetDepth(subDomain->GetDM(), &depth) >> checkError;
    GetRange(depth, cellIS, cStart, cEnd, cells);
}

void ablate::finiteVolume::FiniteVolume::GetFaceRange(IS& faceIS, PetscInt& fStart, PetscInt& fEnd, const PetscInt*& faces) {
    // Start out getting all of the cells
    PetscInt depth;
    DMPlexGetDepth(subDomain->GetDM(), &depth) >> checkError;
    GetRange(depth - 1, faceIS, fStart, fEnd, faces);
}

void ablate::finiteVolume::FiniteVolume::GetRange(PetscInt depth, IS& pointIS, PetscInt& pStart, PetscInt& pEnd, const PetscInt*& points) {
    // Start out getting all of the points
    IS allPointIS;
    DMGetStratumIS(subDomain->GetDM(), "dim", depth, &allPointIS) >> checkError;
    if (!allPointIS) {
        DMGetStratumIS(subDomain->GetDM(), "depth", depth, &allPointIS) >> checkError;
    }

    // If there is a label for this region, get only the parts of the mesh that here
    const auto label = GetSubDomain().GetLabel();
    if (label) {
        IS labelIS;
        DMLabelGetStratumIS(label, GetRegion()->GetValues().front(), &labelIS) >> checkError;
        ISIntersect_Caching_Internal(allPointIS, labelIS, &pointIS) >> checkError;
        ISDestroy(&labelIS) >> checkError;
    } else {
        PetscObjectReference((PetscObject)allPointIS) >> checkError;
        pointIS = allPointIS;
    }

    // Get the point range
    ISGetPointRange(pointIS, &pStart, &pEnd, &points) >> checkError;

    // Clean up the allCellIS
    ISDestroy(&allPointIS) >> checkError;
}

void ablate::finiteVolume::FiniteVolume::RestoreRange(IS& pointIS, PetscInt& pStart, PetscInt& pEnd, const PetscInt*& points) {
    ISRestorePointRange(pointIS, &pStart, &pEnd, &points) >> checkError;
    ISDestroy(&pointIS) >> checkError;
}

#include "parser/registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::finiteVolume::FiniteVolume, "finite volume solver", ARG(std::string, "id", "the name of the flow field"),
         OPT(domain::Region, "region", "the region to apply this solver.  Default is entire domain"), OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"),
         ARG(std::vector<ablate::domain::FieldDescriptor>, "fields", "field descriptions"),
         ARG(std::vector<ablate::finiteVolume::processes::Process>, "processes", "the processes used to describe the flow"),
         OPT(std::vector<mathFunctions::FieldFunction>, "initialization", "the flow field initialization"),
         OPT(std::vector<finiteVolume::boundaryConditions::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"),
         OPT(std::vector<mathFunctions::FieldFunction>, "exactSolution", "optional exact solutions that can be used for error calculations"));