#include "finiteVolumeSolver.hpp"
#include <petsc/private/dmpleximpl.h>
#include <utilities/mpiError.hpp>
#include <utilities/petscError.hpp>
#include "processes/process.hpp"

ablate::finiteVolume::FiniteVolumeSolver::FiniteVolumeSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options,
                                                             std::vector<std::shared_ptr<processes::Process>> processes, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization,
                                                             std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                             std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolution)
    : Solver(std::move(solverId), std::move(region), std::move(options)),
      processes(std::move(processes)),
      initialization(std::move(initialization)),
      boundaryConditions(std::move(boundaryConditions)),
      exactSolutions(std::move(exactSolution)) {}

ablate::finiteVolume::FiniteVolumeSolver::~FiniteVolumeSolver() {}

void ablate::finiteVolume::FiniteVolumeSolver::Setup() {
    // march over process and link to the flow
    for (const auto& process : processes) {
        process->Initialize(*this);
    }

    // Set the flux calculator solver for each component
    PetscDSSetFromOptions(subDomain->GetDiscreteSystem()) >> checkError;
}

void ablate::finiteVolume::FiniteVolumeSolver::Initialize() {
    // add each boundary condition
    for (auto boundary : boundaryConditions) {
        const auto& fieldId = subDomain->GetField(boundary->GetFieldName());

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

PetscErrorCode ablate::finiteVolume::FiniteVolumeSolver::ComputeRHSFunction(PetscReal time, Vec locXVec, Vec locFVec) {
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
        ComputeFlux(time, locXVec, subDomain->GetAuxVector(), locFVec);
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

void ablate::finiteVolume::FiniteVolumeSolver::RegisterRHSFunction(FVMRHSFluxFunction function, void* context, std::string field, std::vector<std::string> inputFields,
                                                                   std::vector<std::string> auxFields) {
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

void ablate::finiteVolume::FiniteVolumeSolver::RegisterRHSFunction(FVMRHSPointFunction function, void* context, std::vector<std::string> fields, std::vector<std::string> inputFields,
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

void ablate::finiteVolume::FiniteVolumeSolver::RegisterRHSFunction(RHSArbitraryFunction function, void* context) { rhsArbitraryFunctions.push_back(std::make_pair(function, context)); }

void ablate::finiteVolume::FiniteVolumeSolver::RegisterAuxFieldUpdate(AuxFieldUpdateFunction function, void* context, std::string auxField, std::vector<std::string> inputFields) {
    // find the field location
    auto& auxFieldLocation = subDomain->GetField(auxField);

    AuxFieldUpdateFunctionDescription functionDescription{.function = function, .context = context, .inputFields = {}, .auxField = auxFieldLocation.id};

    for (std::size_t i = 0; i < inputFields.size(); i++) {
        auto fieldId = subDomain->GetField(inputFields[i]);
        functionDescription.inputFields.push_back(fieldId.id);
    }

    auxFieldUpdateFunctionDescriptions.push_back(functionDescription);
}

void ablate::finiteVolume::FiniteVolumeSolver::ComputeTimeStep(TS ts, ablate::solver::Solver& solver) {
    auto& flowFV = static_cast<ablate::finiteVolume::FiniteVolumeSolver&>(solver);
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

void ablate::finiteVolume::FiniteVolumeSolver::RegisterComputeTimeStepFunction(ComputeTimeStepFunction function, void* ctx) { timeStepFunctions.push_back(std::make_pair(function, ctx)); }
void ablate::finiteVolume::FiniteVolumeSolver::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) const {
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

void ablate::finiteVolume::FiniteVolumeSolver::UpdateAuxFields(PetscReal time, Vec locXVec, Vec locAuxField) {
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



void ablate::finiteVolume::FiniteVolumeSolver::ComputeFlux(PetscReal time, Vec locXVec, Vec locAuxField, Vec locF) {
    auto dm = subDomain->GetDM();
    // Get the cell range
    IS cellIS;
    PetscInt cStart, cEnd;
    const PetscInt* cells;
    GetCellRange(cellIS, cStart, cEnd, cells);

    /* 1: Get sizes from dm and dmAux */
    PetscSection section = nullptr;
    DMGetLocalSection(dm, &section) >> checkError;

    DMLabel ghostLabel;
    DMGetLabel(dm, "ghost", &ghostLabel) >> checkError;

    // Get the ds from he subDomain and required info
    auto ds = subDomain->GetDiscreteSystem();
    PetscInt nf, totDim;
    PetscDSGetNumFields(ds, &nf) >> checkError;
    PetscDSGetTotalDimension(ds, &totDim) >> checkError;
    PetscInt dim = subDomain->GetDimensions();

    // Check to see if the dm has an auxVec/auxDM associated with it.  If it does, extract it
    PetscDS dsAux = subDomain->GetAuxDiscreteSystem();
    PetscInt naf =0, totDimAux =0;
    if (locAuxField) {
        PetscDSGetTotalDimension(dsAux, &totDimAux) >> checkError;
        PetscDSGetNumFields(dsAux, &naf) >> checkError;
    }

    /* 2: Get geometric data */
    // We can use a single call for the geometry data because it does not depend on the fv object
    Vec cellGeometryFVM = NULL, faceGeometryFVM = NULL;
    const PetscScalar* cellGeomArray = NULL;
    const PetscScalar* faceGeomArray = NULL;
    DMPlexGetGeometryFVM(dm, &faceGeometryFVM, &cellGeometryFVM, NULL) >> checkError;
    VecGetArrayRead(faceGeometryFVM, &cellGeomArray) >> checkError;
    VecGetArrayRead(cellGeometryFVM, &faceGeomArray) >> checkError;
    DM faceDM, cellDM;
    VecGetDM(faceGeometryFVM, &faceDM)>> checkError;
    VecGetDM(cellGeometryFVM, &cellDM)>> checkError;

    // there must be a separate gradient vector/dm for field because they can be different sizes
    std::vector<DM> dmGrads(nf, nullptr);
    std::vector<Vec> locGradVecs(nf, nullptr);
    std::vector<DM> dmAuxGrads(naf, nullptr);
    std::vector<Vec> locAuxGradVecs(naf, nullptr);

    /* Reconstruct and limit cell gradients */
    // for each field compute the gradient in the localGrads vector
    for (const auto& field : subDomain->GetFields()) {
        ComputeFieldGradients(field, cellGeometryFVM, faceGeometryFVM, locXVec, locGradVecs[field.subId], dmGrads[field.subId] );
    }

    // do the same for the aux fields
    for (const auto& field : subDomain->GetFields(domain::FieldLocation::AUX)) {
        ComputeFieldGradients(field, cellGeometryFVM, faceGeometryFVM, locAuxField, locAuxGradVecs[field.subId], dmAuxGrads[field.subId] );

        auto fvm = (PetscFV)subDomain->GetPetscFieldObject(field);
        ABLATE_FillGradientBoundary(dm, fvm,locAuxField, locAuxGradVecs[field.subId] ) >> checkError;
    }

    // Get raw access to the computed values
    const PetscScalar *xArray, *auxArray = nullptr;
    VecGetArrayRead(locXVec, &xArray) >> checkError;
    if(locAuxField){
        VecGetArrayRead(locAuxField, &auxArray) >> checkError;
    }

    std::vector<const PetscScalar*> locGradArrays(nf, nullptr);
    std::vector<const PetscScalar*> locAuxGradArrays(naf, nullptr);
    for (const auto& field : subDomain->GetFields()) {
        if(locGradVecs[field.subId]){
            VecGetArrayRead(locGradVecs[field.subId], &locGradArrays[field.subId]) >> checkError;
        }
    }
    for (const auto& field : subDomain->GetFields(domain::FieldLocation::AUX)) {
        if(locAuxGradVecs[field.subId]){
            VecGetArrayRead(locAuxGradVecs[field.subId], &locAuxGradArrays[field.subId]) >> checkError;
        }
    }

    // get raw access to the locF
    PetscScalar *locFArray;
    VecGetArray(locF, &locFArray) >> checkError;

    // Size up the work arrays (uL, uR, gradL, gradR, auxL, auxR, gradAuxL, gradAuxR), these are only sized for one face at a time
    PetscScalar     *flux;
    DMGetWorkArray(dm, totDim, MPIU_SCALAR, &flux) >> checkError;

    PetscScalar *uL, *uR;
    DMGetWorkArray(dm, totDim, MPIU_SCALAR, &uL) >> checkError;
    DMGetWorkArray(dm, totDim, MPIU_SCALAR, &uR) >> checkError;

    PetscScalar *gradL, *gradR;
    DMGetWorkArray(dm, dim*totDim, MPIU_SCALAR, &gradL) >> checkError;
    DMGetWorkArray(dm, dim*totDim, MPIU_SCALAR, &gradR) >> checkError;

    // size up the aux variables
    PetscScalar *auxL = nullptr, *auxR = nullptr;
    PetscScalar *gradAuxL = nullptr, *gradAuxR = nullptr;
    if(auto dmAux = subDomain->GetAuxDM()){
        DMGetWorkArray(dmAux, totDimAux, MPIU_SCALAR, &auxL) >> checkError;
        DMGetWorkArray(dmAux, totDimAux, MPIU_SCALAR, &uR) >> checkError;

        DMGetWorkArray(dmAux, dim*totDimAux, MPIU_SCALAR, &gradAuxR) >> checkError;
        DMGetWorkArray(dmAux, dim*totDimAux, MPIU_SCALAR, &gradAuxL) >> checkError;
    }

    // Precompute the offsets to pass into the rhsFluxFunctionDescriptions
    std::vector<PetscInt> fluxComponentSize (rhsFluxFunctionDescriptions.size());
    std::vector<PetscInt> fluxSubId (rhsFluxFunctionDescriptions.size());
    std::vector<std::vector<PetscInt>> uOff (rhsFluxFunctionDescriptions.size());
    std::vector<std::vector<PetscInt>> aOff (rhsFluxFunctionDescriptions.size());
    std::vector<std::vector<PetscInt>> uOff_x (rhsFluxFunctionDescriptions.size());
    std::vector<std::vector<PetscInt>> aOff_x (rhsFluxFunctionDescriptions.size());

    // Get the full set of offsets from the ds
    PetscInt * uOffTotal;
    PetscInt * uGradOffTotal;
    PetscDSGetComponentOffsets(ds, &uOffTotal)>> checkError;
    PetscDSGetComponentDerivativeOffsets(ds, &uGradOffTotal)>> checkError;

    for (std::size_t fun = 0; fun < rhsFluxFunctionDescriptions.size(); fun++ ){
        const auto& field = subDomain->GetField(rhsFluxFunctionDescriptions[fun].field);
        fluxComponentSize[fun] = field.numberComponents;
        fluxSubId[fun] = field.subId;
        for (PetscInt f =0; f < rhsFluxFunctionDescriptions[fun].numberInputFields; f++){
            uOff[fun].push_back(uOffTotal[rhsFluxFunctionDescriptions[fun].inputFields[f]]);
            uOff_x[fun].push_back(uGradOffTotal[rhsFluxFunctionDescriptions[fun].inputFields[f]]);
        }
    }

    if (dsAux) {
        PetscInt* auxOffTotal;
        PetscInt* auxGradOffTotal;
        PetscDSGetComponentOffsets(dsAux, &auxOffTotal) >> checkError;
        PetscDSGetComponentDerivativeOffsets(dsAux, &auxGradOffTotal) >> checkError;
        for (std::size_t fun = 0; fun < rhsFluxFunctionDescriptions.size(); fun++ ){
            for (PetscInt f =0; f < rhsFluxFunctionDescriptions[fun].numberAuxFields; f++){
                aOff[fun].push_back(auxOffTotal[rhsFluxFunctionDescriptions[fun].auxFields[f]]);
                aOff_x[fun].push_back(auxGradOffTotal[rhsFluxFunctionDescriptions[fun].auxFields[f]]);
            }
        }
    }

    // March over each face in this region
    IS faceIS;
    PetscInt fStart, fEnd;
    const PetscInt* faces;
    GetFaceRange(faceIS, fStart, fEnd, faces);
    for (PetscInt f = fStart; f < fEnd; ++f) {
        const PetscInt face = faces ? faces[f] : f;

        // make sure that this is a valid face
        PetscInt ghost, nsupp, nchild;
        DMLabelGetValue(ghostLabel, face, &ghost)>> checkError;
        DMPlexGetSupportSize(dm, face, &nsupp)>> checkError;
        DMPlexGetTreeChildren(dm, face, &nchild, NULL)>> checkError;
        if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;

        // Get the face geometry
        const PetscInt *faceCells;
        PetscFVFaceGeom *fg;
        PetscFVCellGeom *cgL, *cgR;
        DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg)>> checkError;
        DMPlexGetSupport(dm, face, &faceCells)>> checkError;
        DMPlexPointLocalRead(cellDM, faceCells[0], cellGeomArray, &cgL)>> checkError;
        DMPlexPointLocalRead(cellDM, faceCells[1], cellGeomArray, &cgR)>> checkError;

        // compute the left/right face values
        ProjectToFace(subDomain->GetFields(), ds, *fg, faceCells[0], *cgL,dm, xArray, dmGrads, locGradArrays, uL, gradL);
        ProjectToFace(subDomain->GetFields(), ds, *fg, faceCells[1], *cgR,dm, xArray, dmGrads, locGradArrays, uR, gradR);

        // determine the left/right cells
        if(auxArray) {
            ProjectToFace(subDomain->GetFields(domain::FieldLocation::AUX), dsAux, *fg, faceCells[0], *cgL, dm, auxArray, dmAuxGrads, locAuxGradArrays, auxL, gradAuxL);
            ProjectToFace(subDomain->GetFields(domain::FieldLocation::AUX), dsAux, *fg, faceCells[1], *cgR, dm, auxArray, dmAuxGrads, locAuxGradArrays, gradAuxR, gradAuxR);
        }

        // March over each source function
        for(std::size_t fun =0; fun < rhsFluxFunctionDescriptions.size(); fun++){
            const auto& rhsFluxFunctionDescription = rhsFluxFunctionDescriptions[fun];
            rhsFluxFunctionDescription.function(dim, fg, &uOff[fun][0], &uOff_x[fun][0], uL, uR, gradL, gradR, &aOff[fun][0], &aOff_x[fun][0], auxL, auxR, gradAuxL, gradAuxR, flux, rhsFluxFunctionDescription.context) >> checkError;

            // add the flux back to the cell
            PetscScalar    *fL = NULL, *fR = NULL;
            DMLabelGetValue(ghostLabel,faceCells[0],&ghost) >> checkError;
            if (ghost <= 0) {DMPlexPointLocalFieldRef(dm, faceCells[0], fluxSubId[fun], locFArray, &fL) >> checkError;}
            DMLabelGetValue(ghostLabel,faceCells[1],&ghost) >> checkError;
            if (ghost <= 0) {DMPlexPointLocalFieldRef(dm, faceCells[1], fluxSubId[fun], locFArray, &fR) >> checkError;}

            for (PetscInt d = 0; d < fluxComponentSize[fun]; ++d) {
                if (fL) fL[d] -= flux[d]/cgL->volume;
                if (fR) fR[d] += flux[d]/cgR->volume;
            }
        }


        // project the auxFields to the face

        std::cout << "face (" << f << "): " << face << std::endl;
    }


    // cleanup
    DMRestoreWorkArray(dm, totDim, MPIU_SCALAR, &flux) >> checkError;
    DMRestoreWorkArray(dm, totDim, MPIU_SCALAR, &uL) >> checkError;
    DMRestoreWorkArray(dm, totDim, MPIU_SCALAR, &uR) >> checkError;
    DMRestoreWorkArray(dm, dim*totDim, MPIU_SCALAR, &gradL) >> checkError;
    DMRestoreWorkArray(dm, dim*totDim, MPIU_SCALAR, &gradL) >> checkError;

    if(auto dmAux = subDomain->GetAuxDM()){
        DMRestoreWorkArray(dmAux, totDimAux, MPIU_SCALAR, &auxL) >> checkError;
        DMRestoreWorkArray(dmAux, totDimAux, MPIU_SCALAR, &uR) >> checkError;

        DMRestoreWorkArray(dmAux, dim*totDimAux, MPIU_SCALAR, &gradAuxR) >> checkError;
        DMRestoreWorkArray(dmAux, dim*totDimAux, MPIU_SCALAR, &gradAuxL) >> checkError;
    }

    // cleanup (restore access to locGradVecs, locAuxGradVecs with DMRestoreLocalVector)
    VecRestoreArrayRead(locXVec, &xArray) >> checkError;
    if(locAuxField){
        VecRestoreArrayRead(locAuxField, &auxArray) >> checkError;
    }
    for (const auto& field : subDomain->GetFields()) {
        if(locGradVecs[field.subId]){
            VecRestoreArrayRead(locGradVecs[field.subId], &locGradArrays[field.subId]) >> checkError;
            DMRestoreLocalVector(dmGrads[field.subId], &locGradVecs[field.subId]) >> checkError;
        }
    }
    for (const auto& field : subDomain->GetFields(domain::FieldLocation::AUX)) {
        if(locAuxGradVecs[field.subId]){
            VecRestoreArrayRead(locAuxGradVecs[field.subId], &locAuxGradArrays[field.subId]) >> checkError;
            DMRestoreLocalVector(dmAuxGrads[field.subId], &locAuxGradVecs[field.subId]) >> checkError;
        }
    }

    VecRestoreArray(locF, &locFArray) >> checkError;


    RestoreRange(cellIS, cStart, cEnd, cells);
    RestoreRange(faceIS, fStart, fEnd, faces);
    VecRestoreArrayRead(faceGeometryFVM, (const PetscScalar**)&cellGeomArray) >> checkError;
    VecRestoreArrayRead(cellGeometryFVM, (const PetscScalar**)&faceGeomArray) >> checkError;
}

void ablate::finiteVolume::FiniteVolumeSolver::GetCellRange(IS& cellIS, PetscInt& cStart, PetscInt& cEnd, const PetscInt*& cells) {
    // Start out getting all of the cells
    PetscInt depth;
    DMPlexGetDepth(subDomain->GetDM(), &depth) >> checkError;
    GetRange(depth, cellIS, cStart, cEnd, cells);
}

void ablate::finiteVolume::FiniteVolumeSolver::GetFaceRange(IS& faceIS, PetscInt& fStart, PetscInt& fEnd, const PetscInt*& faces) {
    // Start out getting all of the cells
    PetscInt depth;
    DMPlexGetDepth(subDomain->GetDM(), &depth) >> checkError;
    GetRange(depth - 1, faceIS, fStart, fEnd, faces);
}

void ablate::finiteVolume::FiniteVolumeSolver::GetRange(PetscInt depth, IS& pointIS, PetscInt& pStart, PetscInt& pEnd, const PetscInt*& points) {
    // Start out getting all of the points
    IS allPointIS;
    DMGetStratumIS(subDomain->GetDM(), "dim", depth, &allPointIS) >> checkError;
    if (!allPointIS) {
        DMGetStratumIS(subDomain->GetDM(), "depth", depth, &allPointIS) >> checkError;
    }

    // If there is a label for this solver, get only the parts of the mesh that here
    if (const auto& region = GetRegion()) {
        DMLabel label;
        DMGetLabel(subDomain->GetDM(), region->GetName().c_str(), &label);

        IS labelIS;
        DMLabelGetStratumIS(label, GetRegion()->GetValue(), &labelIS) >> checkError;
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

void ablate::finiteVolume::FiniteVolumeSolver::RestoreRange(IS& pointIS, PetscInt& pStart, PetscInt& pEnd, const PetscInt*& points) {
    ISRestorePointRange(pointIS, &pStart, &pEnd, &points) >> checkError;
    ISDestroy(&pointIS) >> checkError;
}

void ablate::finiteVolume::FiniteVolumeSolver::ComputeFieldGradients(const ablate::domain::Field& field, Vec cellGeometryVec, Vec faceGeometryVec, Vec xGlobVec, Vec& gradLocVec, DM& dmGrad) {
    // get the FVM petsc field associated with this field
    auto fvm = (PetscFV)subDomain->GetPetscFieldObject(field);
    auto dm = subDomain->GetFieldDM(field);

    // Get the dm for this grad field
    DMPlexGetDataFVM_MulfiField(dm, fvm, NULL, NULL, &dmGrad) >> checkError;

    // Create a gradLocVec
    DMGetLocalVector(dmGrad, &gradLocVec)>> checkError;

    // Get the correct sized vec (gradient for this field)
    Vec gradGlobVec;
    DMGetGlobalVector(dmGrad, &gradGlobVec) >> checkError;

    // check to see if there is a ghost label
    DMLabel ghostLabel;
    DMGetLabel(dm, "ghost", &ghostLabel)>> checkError;

    // Get the face geometry
    DM dmFace;
    const PetscScalar *faceGeometryArray;
    VecGetDM(faceGeometryVec, &dmFace)>> checkError;
    VecGetArrayRead(faceGeometryVec, &faceGeometryArray);

    // extract the global x array
    const PetscScalar *xGlobArray;
    VecGetArrayRead(xGlobVec, &xGlobArray);

    // extract the global grad array
    PetscScalar *gradGlobArray;
    VecGetArray(gradGlobVec, &gradGlobArray);

    // March over only the faces in region
    IS faceIS;
    PetscInt fStart, fEnd;
    const PetscInt* faces;
    GetFaceRange(faceIS, fStart, fEnd, faces);

    // Get the dof and dim
    PetscInt dim = subDomain->GetDimensions();
    PetscInt dof = field.numberComponents;

    for (PetscInt f = fStart; f < fEnd; ++f) {
        PetscInt face = faces? faces[f] : f;

        // make sure that this is a face we should use
        PetscBool boundary;
        PetscInt ghost = -1;
        if(ghostLabel){
            DMLabelGetValue(ghostLabel, face, &ghost);
        }
        DMIsBoundaryPoint(dm, face, &boundary);
        PetscInt numChildren;
        DMPlexGetTreeChildren(dm, face, &numChildren, NULL);
        if (ghost >= 0 || boundary || numChildren) continue;

        // Do a sanity check on the number of cells connected to this face
        PetscInt numCells;
        DMPlexGetSupportSize(dm, face, &numCells);
        if (numCells != 2) {
            throw std::runtime_error("face " + std::to_string(face) +  " has " + std::to_string(numCells) +" support points (cells): expected 2");
        }

        // add in the contribuations from this face
        const PetscInt* cells;
        PetscFVFaceGeom* fg;
        PetscScalar* cx[2];
        PetscScalar* cgrad[2];

        DMPlexGetSupport(dm, face, &cells);
        DMPlexPointLocalRead(dmFace, face, faceGeometryArray, &fg);
        for (PetscInt c = 0; c < 2; ++c) {
            DMPlexPointLocalFieldRead(dm, cells[c], field.id, xGlobArray, &cx[c]);
            DMPlexPointGlobalRef(dmGrad, cells[c], gradGlobArray, &cgrad[c]);
        }
        for (PetscInt pd = 0; pd < dof; ++pd) {
            PetscScalar delta = cx[1][pd] - cx[0][pd];

            for (PetscInt d = 0; d < dim; ++d) {
                if (cgrad[0]) cgrad[0][pd * dim + d] += fg->grad[0][d] * delta;
                if (cgrad[1]) cgrad[1][pd * dim + d] -= fg->grad[1][d] * delta;
            }
        }
    }

    //TODO: add back in limiters

    // Communicate gradient values
    VecRestoreArray(gradGlobVec, &gradGlobArray) >> checkError;
    DMGlobalToLocalBegin(dmGrad, gradGlobVec, INSERT_VALUES, gradLocVec) >> checkError;
    DMGlobalToLocalEnd(dmGrad, gradGlobVec, INSERT_VALUES, gradLocVec) >> checkError;

    // cleanup
    VecRestoreArrayRead(xGlobVec, &xGlobArray) >> checkError;
    VecRestoreArrayRead(faceGeometryVec, &faceGeometryArray) >> checkError;
    RestoreRange(faceIS, fStart, fEnd, faces);
    DMRestoreGlobalVector(dmGrad, &gradGlobVec)>> checkError;
}
void ablate::finiteVolume::FiniteVolumeSolver::ProjectToFace(const std::vector<domain::Field>& fields, PetscDS ds, const PetscFVFaceGeom& faceGeom, PetscInt cellId, const PetscFVCellGeom& cellGeom, DM dm, const PetscScalar* xArray,
                                                             const std::vector<DM>& dmGrads, const std::vector<const PetscScalar*>& gradArrays, PetscScalar* u, PetscScalar* grad, bool projectField) {

    const auto dim = subDomain->GetDimensions();

    // Keep track of derivative offset
    PetscInt *offsets;
    PetscInt *dirOffsets;
    PetscDSGetComponentOffsets(ds, &offsets) >> checkError;
    PetscDSGetComponentDerivativeOffsets(ds, &dirOffsets) >> checkError;

    // March over each field
    for(const auto& field: fields){
        PetscReal dx[3];
        PetscScalar *xCell;
        PetscScalar *gradCell;

        // Get the field values at this cell
        DMPlexPointLocalFieldRead(dm, cellId, field.subId, xArray, &xCell) >> checkError;

        // If we need to project the field
        if(projectField && dmGrads[field.subId]){
            DMPlexPointLocalRead(dmGrads[field.subId], cellId, gradArrays[field.subId], &gradCell) >> checkError;
            DMPlex_WaxpyD_Internal(dim, -1, cellGeom.centroid, faceGeom.centroid, dx);

            // Project the cell centered value onto the face
            for (PetscInt c = 0; c < field.numberComponents; ++c) {
                u[offsets[field.subId] + c] = xCell[c] + DMPlex_DotD_Internal(dim, &gradCell[c * dim], dx);

                // copy the gradient into the grad vector
                for (PetscInt d = 0; d < dim; d++) {
                    grad[dirOffsets[field.subId] + c * dim + d] = gradCell[c * dim + d];
                }
            }

        }else if (dmGrads[field.subId]) {
            // Project the cell centered value onto the face
            DMPlexPointLocalRead(dmGrads[field.subId], cellId, gradArrays[field.subId], &gradCell) >> checkError;
            // Project the cell centered value onto the face
            for (PetscInt c = 0; c < field.numberComponents; ++c) {
                u[offsets[field.subId] + c] = xCell[c];

                // copy the gradient into the grad vector
                for (PetscInt d = 0; d < dim; d++) {
                    grad[dirOffsets[field.subId] + c * dim + d] = gradCell[c * dim + d];
                }
            }

        }else{
            // Just copy the cell centered value on to the face
            for (PetscInt c = 0; c < field.numberComponents; ++c) {
                u[offsets[field.subId] + c] = xCell[c];

                // fill the grad with NAN to prevent use
                for (PetscInt d = 0; d < dim; d++) {
                    grad[dirOffsets[field.subId] + c * dim + d] = NAN;
                }
            }
        }



    }


}

#include "parser/registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::finiteVolume::FiniteVolumeSolver, "finite volume solver", ARG(std::string, "id", "the name of the flow field"),
         OPT(domain::Region, "region", "the region to apply this solver.  Default is entire domain"), OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"),
         ARG(std::vector<ablate::finiteVolume::processes::Process>, "processes", "the processes used to describe the flow"),
         OPT(std::vector<mathFunctions::FieldFunction>, "initialization", "the flow field initialization"),
         OPT(std::vector<finiteVolume::boundaryConditions::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"),
         OPT(std::vector<mathFunctions::FieldFunction>, "exactSolution", "optional exact solutions that can be used for error calculations"));