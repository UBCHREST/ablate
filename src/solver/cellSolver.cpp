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
    IS cellIS;
    PetscInt cStart, cEnd;
    const PetscInt* cells;
    GetCellRange(cellIS, cStart, cEnd, cells);

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
    for (PetscInt c = cStart; c < cEnd; ++c) {
        PetscFVCellGeom* cellGeom;
        const PetscReal* fieldValues;
        PetscReal* auxValues;

        // Get the cell location
        const PetscInt cell = cells ? cells[c] : c;

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

    RestoreRange(cellIS, cStart, cEnd, cells);

    DMDestroy(&plex) >> checkError;
}

void ablate::solver::CellSolver::Setup() {
    // Compute the dm geometry
    DMPlexComputeGeometryFVM(subDomain->GetDM(), &cellGeomVec, &faceGeomVec) >> checkError;
}