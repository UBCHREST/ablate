#include "cellSolver.hpp"

#include <utility>

ablate::solver::CellSolver::CellSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options)
    : Solver(std::move(solverId), std::move(region), std::move(options)) {}

ablate::solver::CellSolver::~CellSolver() {
    if (cellGeomVec) {
        VecDestroy(&cellGeomVec) >> checkError;
    }
    if (faceGeomVec) {
        VecDestroy(&faceGeomVec) >> checkError;
    }
    if (solverRegionMinusGhost) {
        ISDestroy(&solverRegionMinusGhost) >> checkError;
    }
}

void ablate::solver::CellSolver::RegisterAuxFieldUpdate(ablate::solver::CellSolver::AuxFieldUpdateFunction function, void* context, const std::vector<std::string>& auxFields,
                                                        const std::vector<std::string>& inputFields) {
    AuxFieldUpdateFunctionDescription functionDescription{.function = function, .context = context, .inputFields = {}, .auxFields = {}};

    for (const auto& auxField : auxFields) {
        auto fieldId = subDomain->GetField(auxField);
        functionDescription.auxFields.push_back(fieldId.id);
    }

    for (const auto& inputField : inputFields) {
        auto fieldId = subDomain->GetField(inputField);
        functionDescription.inputFields.push_back(fieldId.id);
    }

    // Don't add the same field more than once
    auto location = std::find_if(auxFieldUpdateFunctionDescriptions.begin(), auxFieldUpdateFunctionDescriptions.end(), [&functionDescription](const auto& description) {
        return functionDescription.auxFields == description.auxFields;
    });

    if (location == auxFieldUpdateFunctionDescriptions.end()) {
        auxFieldUpdateFunctionDescriptions.push_back(functionDescription);
    } else {
        *location = functionDescription;
    }
}
void ablate::solver::CellSolver::UpdateAuxFields(PetscReal time, Vec locXVec, Vec locAuxField) {
    DM plex;
    DM auxDM = GetSubDomain().GetAuxDM();
    // Convert to a dmplex
    DMConvert(GetSubDomain().GetDM(), DMPLEX, &plex) >> checkError;

    // Get the valid cell range over this region
    solver::Range cellRange;
    GetCellRange(cellRange);

    // Extract the cell geometry, and the dm that holds the information
    DM dmCell;
    const PetscScalar* cellGeomArray;
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

    PetscInt* uOffTotal;
    PetscDSGetComponentOffsets(subDomain->GetDiscreteSystem(), &uOffTotal) >> checkError;
    PetscInt* aOffTotal;
    PetscDSGetComponentOffsets(subDomain->GetAuxDiscreteSystem(), &aOffTotal) >> checkError;

    // precompute the solution(u) and aux fields
    std::vector<std::vector<PetscInt>> uOff(auxFieldUpdateFunctionDescriptions.size());
    std::vector<std::vector<PetscInt>> aOff(auxFieldUpdateFunctionDescriptions.size());
    for (std::size_t uf = 0; uf < auxFieldUpdateFunctionDescriptions.size(); uf++) {
        for (const auto& inputField : auxFieldUpdateFunctionDescriptions[uf].inputFields) {
            uOff[uf].push_back(uOffTotal[inputField]);
        }
        for (const auto& auxField : auxFieldUpdateFunctionDescriptions[uf].auxFields) {
            aOff[uf].push_back(aOffTotal[auxField]);
        }
    }

    // March over each cell volume
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscFVCellGeom* cellGeom;
        const PetscReal* fieldValues;
        PetscReal* auxValues;

        // Get the cell location
        const PetscInt cell = cellRange.points ? cellRange.points[c] : c;

        DMPlexPointLocalRead(dmCell, cell, cellGeomArray, &cellGeom) >> checkError;
        DMPlexPointLocalRead(plex, cell, locFlowFieldArray, &fieldValues) >> checkError;
        DMPlexPointLocalRead(auxDM, cell, localAuxFlowFieldArray, &auxValues) >> checkError;

        // for each function description
        for (std::size_t uf = 0; uf < auxFieldUpdateFunctionDescriptions.size(); uf++) {
            // If an update function was passed
            auxFieldUpdateFunctionDescriptions[uf].function(time, dim, cellGeom, uOff[uf].data(), fieldValues, aOff[uf].data(), auxValues, auxFieldUpdateFunctionDescriptions[uf].context) >>
                checkError;
        }
    }

    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    VecRestoreArrayRead(locXVec, &locFlowFieldArray) >> checkError;
    VecRestoreArray(locAuxField, &localAuxFlowFieldArray) >> checkError;

    RestoreRange(cellRange);

    DMDestroy(&plex) >> checkError;
}

void ablate::solver::CellSolver::Setup() {
    // Compute the dm geometry
    DMPlexComputeGeometryFVM(subDomain->GetDM(), &cellGeomVec, &faceGeomVec) >> checkError;

    // get the cell is for the solver minus ghost cell
    // Get the original range
    Range cellRange;
    GetCellRange(cellRange);

    // There may be no cells is this solver, so check and return solverRegionMinusGhost
    if (cellRange.start == cellRange.end) {
        solverRegionMinusGhost = nullptr;
        RestoreRange(cellRange);
        return;
    }

    // Get the cell depth
    PetscInt cellDepth;
    DMPlexGetDepth(subDomain->GetDM(), &cellDepth) >> checkError;

    // Get the ghost cell label
    DMLabel ghostLabel;
    DMGetLabel(subDomain->GetDM(), "ghost", &ghostLabel) >> checkError;
    IS ghostIS;
    DMLabelGetStratumIS(ghostLabel, 1, &ghostIS) >> checkError;

    // remove the mpi ghost cells
    IS solverMinusMpiGhost;
    ISDifference(cellRange.is, ghostIS, &solverMinusMpiGhost) >> checkError;
    if (solverMinusMpiGhost) {
        // Now march over each cell and remove it if it is an exterior boundary cell

        PetscInt ghostStart, ghostEnd;
        DMPlexGetGhostCellStratum(subDomain->GetDM(), &ghostStart, &ghostEnd) >> checkError;
        IS bcGhostIS;
        ISCreateStride(PETSC_COMM_SELF, ghostEnd - ghostStart, ghostStart, 1, &bcGhostIS) >> checkError;
        if (bcGhostIS) {
            // remove the bc ghost
            ISDifference(solverMinusMpiGhost, bcGhostIS, &solverRegionMinusGhost) >> checkError;
            ISDestroy(&solverMinusMpiGhost) >> checkError;
            ISDestroy(&bcGhostIS) >> checkError;
        } else {
            // just set the value
            solverRegionMinusGhost = solverMinusMpiGhost;
        }
    }

    // restore the cell range
    RestoreRange(cellRange);
    ISDestroy(&ghostIS) >> checkError;
}

void ablate::solver::CellSolver::GetCellRangeWithoutGhost(Range& faceRange) const {
    // Get the point range
    faceRange.is = solverRegionMinusGhost;
    if (solverRegionMinusGhost == nullptr) {
        // There are no points in this region, so skip
        faceRange.start = 0;
        faceRange.end = 0;
        faceRange.points = nullptr;
    } else {
        // Get the range
        ISGetPointRange(faceRange.is, &faceRange.start, &faceRange.end, &faceRange.points) >> checkError;
        PetscObjectReference((PetscObject)solverRegionMinusGhost) >> checkError;
    }
}