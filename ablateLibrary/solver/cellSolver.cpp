#include "cellSolver.hpp"

#include <utility>

ablate::solver::CellSolver::CellSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options)
    : Solver(std::move(solverId), std::move(region), std::move(options)) {}

void ablate::solver::CellSolver::RegisterAuxFieldUpdate(ablate::solver::CellSolver::AuxFieldUpdateFunction function, void* context, const std::string& auxField,
                                                        const std::vector<std::string>& inputFields) {
    // find the field location
    auto& auxFieldLocation = subDomain->GetField(auxField);

    AuxFieldUpdateFunctionDescription functionDescription{.function = function, .context = context, .inputFields = {}, .auxField = auxFieldLocation.id};

    for (const auto& inputField : inputFields) {
        auto fieldId = subDomain->GetField(inputField);
        functionDescription.inputFields.push_back(fieldId.id);
    }

    // Don't add the same field more than once
    auto location = std::find_if(
        auxFieldUpdateFunctionDescriptions.begin(), auxFieldUpdateFunctionDescriptions.end(), [&auxFieldLocation](const auto& description) { return auxFieldLocation.id == description.auxField; });

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
    Vec cellGeomVec;
    DM dmCell;
    const PetscScalar* cellGeomArray;
    DMPlexGetGeometryFVM(plex, nullptr, &cellGeomVec, nullptr) >> checkError;
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
    PetscInt* uOff = nullptr;
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
