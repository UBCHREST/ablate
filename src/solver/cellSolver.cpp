#include "cellSolver.hpp"
#include <utility>

ablate::solver::CellSolver::CellSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options)
    : Solver(std::move(solverId), std::move(region), std::move(options)) {}

ablate::solver::CellSolver::~CellSolver() {
    if (cellGeomVec) {
        VecDestroy(&cellGeomVec) >> utilities::PetscUtilities::checkError;
    }
    if (faceGeomVec) {
        VecDestroy(&faceGeomVec) >> utilities::PetscUtilities::checkError;
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

void ablate::solver::CellSolver::RegisterSolutionFieldUpdate(ablate::solver::CellSolver::SolutionFieldUpdateFunction function, void* context, const std::vector<std::string>& inputFields) {
    SolutionFieldUpdateFunctionDescription functionDescription{.function = function, .context = context, .inputFieldsOffsets = {}};

    for (const auto& inputField : inputFields) {
        auto fieldId = subDomain->GetField(inputField);
        functionDescription.inputFieldsOffsets.push_back(fieldId.offset);
    }

    // Don't add the same field more than once
    auto location = std::find_if(solutionFieldUpdateFunctionDescriptions.begin(), solutionFieldUpdateFunctionDescriptions.end(), [&functionDescription](const auto& description) {
        return functionDescription.inputFieldsOffsets == description.inputFieldsOffsets;
    });

    if (location == solutionFieldUpdateFunctionDescriptions.end()) {
        solutionFieldUpdateFunctionDescriptions.push_back(functionDescription);
    } else {
        *location = functionDescription;
    }
}

void ablate::solver::CellSolver::UpdateAuxFields(PetscReal time, Vec locXVec, Vec locAuxField) {
    // make sure there are aux fields to update
    if (auxFieldUpdateFunctionDescriptions.empty()) {
        return;
    }

    DM plex;
    DM auxDM = GetSubDomain().GetAuxDM();
    // Convert to a dmplex
    DMConvert(GetSubDomain().GetDM(), DMPLEX, &plex) >> utilities::PetscUtilities::checkError;

    // Get the valid cell range over this region
    ablate::domain::Range cellRange;
    GetCellRange(cellRange);

    // Extract the cell geometry, and the dm that holds the information
    DM dmCell;
    const PetscScalar* cellGeomArray;
    VecGetDM(cellGeomVec, &dmCell) >> utilities::PetscUtilities::checkError;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;

    // extract the low flow and aux fields
    const PetscScalar* locFlowFieldArray;
    VecGetArrayRead(locXVec, &locFlowFieldArray) >> utilities::PetscUtilities::checkError;

    PetscScalar* localAuxFlowFieldArray;
    VecGetArray(locAuxField, &localAuxFlowFieldArray) >> utilities::PetscUtilities::checkError;

    // Get the cell dim
    PetscInt dim = subDomain->GetDimensions();

    // determine the number of fields and the totDim
    PetscInt* uOffTotal;
    PetscDSGetComponentOffsets(subDomain->GetDiscreteSystem(), &uOffTotal) >> utilities::PetscUtilities::checkError;
    PetscInt* aOffTotal;
    PetscDSGetComponentOffsets(subDomain->GetAuxDiscreteSystem(), &aOffTotal) >> utilities::PetscUtilities::checkError;

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

        DMPlexPointLocalRead(dmCell, cell, cellGeomArray, &cellGeom) >> utilities::PetscUtilities::checkError;
        DMPlexPointLocalRead(plex, cell, locFlowFieldArray, &fieldValues) >> utilities::PetscUtilities::checkError;
        DMPlexPointLocalRead(auxDM, cell, localAuxFlowFieldArray, &auxValues) >> utilities::PetscUtilities::checkError;

        // for each function description
        for (std::size_t uf = 0; uf < auxFieldUpdateFunctionDescriptions.size(); uf++) {
            // If an update function was passed
            auxFieldUpdateFunctionDescriptions[uf].function(time, dim, cellGeom, uOff[uf].data(), fieldValues, aOff[uf].data(), auxValues, auxFieldUpdateFunctionDescriptions[uf].context) >>
                utilities::PetscUtilities::checkError;
        }
    }

    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(locXVec, &locFlowFieldArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArray(locAuxField, &localAuxFlowFieldArray) >> utilities::PetscUtilities::checkError;

    RestoreRange(cellRange);

    DMDestroy(&plex) >> utilities::PetscUtilities::checkError;
}

void ablate::solver::CellSolver::UpdateSolutionFields(PetscReal time, Vec globXVec) {
    // make sure there are solution fields to update
    if (solutionFieldUpdateFunctionDescriptions.empty()) {
        return;
    }

    // Get the valid cell range over this region
    ablate::domain::Range cellRange;
    GetCellRange(cellRange);

    // Extract the cell geometry, and the dm that holds the information
    DM dm = subDomain->GetDM();
    DM dmCell;
    const PetscScalar* cellGeomArray;
    VecGetDM(cellGeomVec, &dmCell) >> utilities::PetscUtilities::checkError;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;

    // extract the low flow and aux fields
    PetscScalar* globalFlowFieldArray;
    VecGetArray(globXVec, &globalFlowFieldArray) >> utilities::PetscUtilities::checkError;

    // Get the cell dim
    PetscInt dim = subDomain->GetDimensions();

    // March over each cell volume
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscFVCellGeom* cellGeom;
        PetscReal* fieldValues;

        // Get the cell location
        const PetscInt cell = cellRange.points ? cellRange.points[c] : c;

        DMPlexPointLocalRead(dmCell, cell, cellGeomArray, &cellGeom) >> utilities::PetscUtilities::checkError;
        DMPlexPointGlobalRef(dm, cell, globalFlowFieldArray, &fieldValues) >> utilities::PetscUtilities::checkError;

        // for each function description
        if (fieldValues) {
            for (auto& solutionFieldUpdateFunctionDescription : solutionFieldUpdateFunctionDescriptions) {
                // If an update function was passed
                solutionFieldUpdateFunctionDescription.function(
                    time, dim, cellGeom, solutionFieldUpdateFunctionDescription.inputFieldsOffsets.data(), fieldValues, solutionFieldUpdateFunctionDescription.context) >>
                    utilities::PetscUtilities::checkError;
            }
        }
    }

    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArray(globXVec, &globalFlowFieldArray) >> utilities::PetscUtilities::checkError;

    RestoreRange(cellRange);
}

void ablate::solver::CellSolver::Setup() {
    // Compute the dm geometry
    DMPlexComputeGeometryFVM(subDomain->GetDM(), &cellGeomVec, &faceGeomVec) >> utilities::PetscUtilities::checkError;
}

void ablate::solver::CellSolver::Initialize() {
    // If there are any solution updates
    if (!solutionFieldUpdateFunctionDescriptions.empty()) {
        // Update any solutions fields
        UpdateSolutionFields(0.0, subDomain->GetSolutionVector());

        // register a prestep
        this->RegisterPreStage([this](TS ts, Solver&, PetscReal stageTime) {
            Vec globFlowVec;
            TSGetSolution(ts, &globFlowVec) >> utilities::PetscUtilities::checkError;

            // Update the solution field
            this->UpdateSolutionFields(stageTime, globFlowVec);
        });
    }
}