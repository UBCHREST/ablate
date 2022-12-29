#include "boundarySolverMonitor.hpp"
#include "io/interval/fixedInterval.hpp"

ablate::monitors::BoundarySolverMonitor::~BoundarySolverMonitor() {
    if (boundaryDm) {
        DMDestroy(&boundaryDm) >> utilities::PetscUtilities::checkError;
    }

    if (faceDm) {
        DMDestroy(&faceDm) >> utilities::PetscUtilities::checkError;
    }
}

void ablate::monitors::BoundarySolverMonitor::Register(std::shared_ptr<solver::Solver> solver) {
    Monitor::Register(solver);

    // this monitor will only work boundary solver
    boundarySolver = std::dynamic_pointer_cast<ablate::boundarySolver::BoundarySolver>(solver);
    if (!boundarySolver) {
        throw std::invalid_argument("The BoundarySolverMonitor monitor can only be used with ablate::boundarySolver::BoundarySolver");
    }

    // update the name
    name = solver->GetSolverId() + name;

    // make a copy of the dm for a boundary dm.
    DM coordDM;
    DMGetCoordinateDM(solver->GetSubDomain().GetDM(), &coordDM) >> utilities::PetscUtilities::checkError;
    DMClone(solver->GetSubDomain().GetDM(), &boundaryDm) >> utilities::PetscUtilities::checkError;
    DMSetCoordinateDM(boundaryDm, coordDM) >> utilities::PetscUtilities::checkError;

    // Create a label in the dm copy to mark boundary faces
    DMCreateLabel(boundaryDm, "boundaryFaceLabel") >> utilities::PetscUtilities::checkError;
    DMLabel boundaryFaceLabel;
    DMGetLabel(boundaryDm, "boundaryFaceLabel", &boundaryFaceLabel) >> utilities::PetscUtilities::checkError;

    // Also create a section on each of the faces.  This needs to be a custom section
    PetscSection boundaryFaceSection;
    PetscSectionCreate(PetscObjectComm((PetscObject)boundaryDm), &boundaryFaceSection) >> utilities::PetscUtilities::checkError;
    // Set the max/min bounds
    PetscInt fStart, fEnd;
    DMPlexGetHeightStratum(solver->GetSubDomain().GetDM(), 1, &fStart, &fEnd) >> utilities::PetscUtilities::checkError;
    PetscSectionSetChart(boundaryFaceSection, fStart, fEnd) >> utilities::PetscUtilities::checkError;

    // default section dof to zero
    for (PetscInt f = fStart; f < fEnd; ++f) {
        PetscSectionSetDof(boundaryFaceSection, f, 0) >> utilities::PetscUtilities::checkError;
    }

    // set the label at each of the faces and set the dof at each point
    const auto numberOfComponents = (PetscInt)boundarySolver->GetOutputComponents().size();
    for (const auto& gradientStencil : boundarySolver->GetBoundaryGeometry()) {
        // set both the label (used for filtering) and section for global variable creation
        DMLabelSetValue(boundaryFaceLabel, gradientStencil.geometry.faceId, 1) >> utilities::PetscUtilities::checkError;

        // set the dof at each section to the numberOfComponents
        PetscSectionSetDof(boundaryFaceSection, gradientStencil.geometry.faceId, numberOfComponents) >> utilities::PetscUtilities::checkError;
    }

    // finish the section
    PetscSectionSetUp(boundaryFaceSection) >> utilities::PetscUtilities::checkError;
    DMSetLocalSection(boundaryDm, boundaryFaceSection) >> utilities::PetscUtilities::checkError;
    PetscSectionDestroy(&boundaryFaceSection) >> utilities::PetscUtilities::checkError;

    // Complete the label
    DMPlexLabelComplete(boundaryDm, boundaryFaceLabel) >> utilities::PetscUtilities::checkError;

    // Now create a sub dm with only the faces
    DMPlexFilter(boundaryDm, boundaryFaceLabel, 1, &faceDm) >> utilities::PetscUtilities::checkError;

    // Add each of the output components on each face in the faceDm
    for (const auto& field : boundarySolver->GetOutputComponents()) {
        PetscFV fvm;
        PetscFVCreate(PetscObjectComm(PetscObject(faceDm)), &fvm) >> utilities::PetscUtilities::checkError;
        PetscObjectSetName((PetscObject)fvm, field.c_str()) >> utilities::PetscUtilities::checkError;
        PetscFVSetFromOptions(fvm) >> utilities::PetscUtilities::checkError;
        PetscFVSetNumComponents(fvm, 1) >> utilities::PetscUtilities::checkError;
        PetscInt dim;
        DMGetCoordinateDim(faceDm, &dim) >> utilities::PetscUtilities::checkError;
        PetscFVSetSpatialDimension(fvm, dim) >> utilities::PetscUtilities::checkError;

        DMAddField(faceDm, nullptr, (PetscObject)fvm) >> utilities::PetscUtilities::checkError;
        PetscFVDestroy(&fvm);
    }
    DMCreateDS(faceDm) >> utilities::PetscUtilities::checkError;
}

void ablate::monitors::BoundarySolverMonitor::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    PetscFunctionBeginUser;
    // If this is the first output, store a copy of the faceDm
    if (sequenceNumber == 0) {
        DMView(faceDm, viewer) >> utilities::PetscUtilities::checkError;
    }

    // Set the output sequence number to the face dm
    DMSetOutputSequenceNumber(faceDm, sequenceNumber, time) >> utilities::PetscUtilities::checkError;

    // Create a local version of the solution (X) vector
    Vec locXVec;
    DMGetLocalVector(GetSolver()->GetSubDomain().GetDM(), &locXVec) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocalBegin(GetSolver()->GetSubDomain().GetDM(), GetSolver()->GetSubDomain().GetSolutionVector(), INSERT_VALUES, locXVec) >> utilities::PetscUtilities::checkError;

    // create a local vector on the boundary solver
    Vec localBoundaryVec;
    DMGetLocalVector(boundaryDm, &localBoundaryVec) >> utilities::PetscUtilities::checkError;
    VecZeroEntries(localBoundaryVec) >> utilities::PetscUtilities::checkError;

    // finish with the locXVec
    DMGlobalToLocalEnd(GetSolver()->GetSubDomain().GetDM(), GetSolver()->GetSubDomain().GetSolutionVector(), INSERT_VALUES, locXVec) >> utilities::PetscUtilities::checkError;

    // compute the rhs
    boundarySolver->ComputeRHSFunction(time, locXVec, localBoundaryVec, boundarySolver->GetOutputFunctions()) >> utilities::PetscUtilities::checkError;

    // Create a local vector for just the monitor
    Vec localFaceVec;
    DMGetLocalVector(faceDm, &localFaceVec) >> utilities::PetscUtilities::checkError;
    VecZeroEntries(localFaceVec) >> utilities::PetscUtilities::checkError;

    // Get the raw data for the global vectors
    const PetscScalar* localBoundaryArray;
    VecGetArrayRead(localBoundaryVec, &localBoundaryArray) >> utilities::PetscUtilities::checkError;
    PetscScalar* localFaceArray;
    VecGetArray(localFaceVec, &localFaceArray) >> utilities::PetscUtilities::checkError;

    // Determine the size of data
    PetscInt dataSize;
    VecGetBlockSize(localFaceVec, &dataSize) >> utilities::PetscUtilities::checkError;

    // March over each cell in the face dm
    PetscInt cStart, cEnd;
    DMPlexGetHeightStratum(faceDm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;

    // get the mapping information
    IS faceIs;
    const PetscInt* faceToBoundary = nullptr;
    DMPlexGetSubpointIS(faceDm, &faceIs) >> utilities::PetscUtilities::checkError;
    ISGetIndices(faceIs, &faceToBoundary) >> utilities::PetscUtilities::checkError;

    // Copy over the values that are in the globalFaceVec.  We may skip some local ghost values
    if (localBoundaryArray && localFaceArray) {
        for (PetscInt facePt = cStart; facePt < cEnd; ++facePt) {
            PetscInt boundaryPt = faceToBoundary[facePt];

            const PetscScalar* localBoundaryData = nullptr;
            PetscScalar* globalFaceData = nullptr;

            DMPlexPointLocalRead(boundaryDm, boundaryPt, localBoundaryArray, &localBoundaryData) >> utilities::PetscUtilities::checkError;
            DMPlexPointLocalRef(faceDm, facePt, localFaceArray, &globalFaceData) >> utilities::PetscUtilities::checkError;
            if (globalFaceData && localBoundaryData) {
                PetscArraycpy(globalFaceData, localBoundaryData, dataSize) >> utilities::PetscUtilities::checkError;
            }
        }
    }

    // restore
    ISRestoreIndices(faceIs, &faceToBoundary) >> utilities::PetscUtilities::checkError;

    VecRestoreArrayRead(localBoundaryVec, &localBoundaryArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArray(localFaceVec, &localFaceArray) >> utilities::PetscUtilities::checkError;

    // Map to a global array with add values
    Vec globalFaceVec;
    DMGetGlobalVector(faceDm, &globalFaceVec) >> utilities::PetscUtilities::checkError;
    PetscObjectSetName((PetscObject)globalFaceVec, GetId().c_str()) >> utilities::PetscUtilities::checkError;
    VecZeroEntries(globalFaceVec);
    DMLocalToGlobal(faceDm, localFaceVec, ADD_VALUES, globalFaceVec) >> utilities::PetscUtilities::checkError;

    // write to the output file
    VecView(globalFaceVec, viewer) >> utilities::PetscUtilities::checkError;
    DMRestoreGlobalVector(faceDm, &globalFaceVec) >> utilities::PetscUtilities::checkError;

    // cleanup
    DMRestoreLocalVector(faceDm, &localFaceVec) >> utilities::PetscUtilities::checkError;
    DMRestoreLocalVector(GetSolver()->GetSubDomain().GetDM(), &locXVec) >> utilities::PetscUtilities::checkError;
    DMRestoreLocalVector(boundaryDm, &localBoundaryVec) >> utilities::PetscUtilities::checkError;
    PetscFunctionReturnVoid();
}

#include "registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::monitors::Monitor, ablate::monitors::BoundarySolverMonitor, "Outputs any provided information from the boundary time to the serializer.");
