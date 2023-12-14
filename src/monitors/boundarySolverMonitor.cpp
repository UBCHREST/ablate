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
    DMPlexFilter(boundaryDm, boundaryFaceLabel, 1, PETSC_FALSE, PETSC_FALSE, NULL, &faceDm) >> utilities::PetscUtilities::checkError;

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

PetscErrorCode ablate::monitors::BoundarySolverMonitor::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    PetscFunctionBeginUser;
    // If this is the first output, store a copy of the faceDm
    if (sequenceNumber == 0) {
        PetscCall(DMView(faceDm, viewer));
    }

    // Set the output sequence number to the face dm
    PetscCall(DMSetOutputSequenceNumber(faceDm, sequenceNumber, time));

    // Create a local version of the solution (X) vector
    Vec locXVec;
    PetscCall(DMGetLocalVector(GetSolver()->GetSubDomain().GetDM(), &locXVec));
    PetscCall(DMGlobalToLocalBegin(GetSolver()->GetSubDomain().GetDM(), GetSolver()->GetSubDomain().GetSolutionVector(), INSERT_VALUES, locXVec));

    // create a local vector on the boundary solver
    Vec localBoundaryVec;
    PetscCall(DMGetLocalVector(boundaryDm, &localBoundaryVec));
    PetscCall(VecZeroEntries(localBoundaryVec));

    // finish with the locXVec
    PetscCall(DMGlobalToLocalEnd(GetSolver()->GetSubDomain().GetDM(), GetSolver()->GetSubDomain().GetSolutionVector(), INSERT_VALUES, locXVec));

    // compute the rhs
    PetscCall(boundarySolver->ComputeRHSFunction(time, locXVec, localBoundaryVec, boundarySolver->GetOutputFunctions()));

    // Create a local vector for just the monitor
    Vec localFaceVec;
    PetscCall(DMGetLocalVector(faceDm, &localFaceVec));
    PetscCall(VecZeroEntries(localFaceVec));

    // Get the raw data for the global vectors
    const PetscScalar* localBoundaryArray;
    PetscCall(VecGetArrayRead(localBoundaryVec, &localBoundaryArray));
    PetscScalar* localFaceArray;
    PetscCall(VecGetArray(localFaceVec, &localFaceArray));

    // Determine the size of data
    PetscInt dataSize;
    PetscCall(VecGetBlockSize(localFaceVec, &dataSize));

    // March over each cell in the face dm
    PetscInt cStart, cEnd;
    PetscCall(DMPlexGetHeightStratum(faceDm, 0, &cStart, &cEnd));

    // get the mapping information
    IS faceIs;
    const PetscInt* faceToBoundary = nullptr;
    PetscCall(DMPlexGetSubpointIS(faceDm, &faceIs));
    PetscCall(ISGetIndices(faceIs, &faceToBoundary));

    // Copy over the values that are in the globalFaceVec.  We may skip some local ghost values
    if (localBoundaryArray && localFaceArray) {
        for (PetscInt facePt = cStart; facePt < cEnd; ++facePt) {
            PetscInt boundaryPt = faceToBoundary[facePt];

            const PetscScalar* localBoundaryData = nullptr;
            PetscScalar* globalFaceData = nullptr;

            PetscCall(DMPlexPointLocalRead(boundaryDm, boundaryPt, localBoundaryArray, &localBoundaryData));
            PetscCall(DMPlexPointLocalRef(faceDm, facePt, localFaceArray, &globalFaceData));
            if (globalFaceData && localBoundaryData) {
                PetscCall(PetscArraycpy(globalFaceData, localBoundaryData, dataSize));
            }
        }
    }

    // restore
    PetscCall(ISRestoreIndices(faceIs, &faceToBoundary));

    PetscCall(VecRestoreArrayRead(localBoundaryVec, &localBoundaryArray));
    PetscCall(VecRestoreArray(localFaceVec, &localFaceArray));

    // Map to a global array with add values
    Vec globalFaceVec;
    PetscCall(DMGetGlobalVector(faceDm, &globalFaceVec));
    PetscCall(PetscObjectSetName((PetscObject)globalFaceVec, GetId().c_str()));
    PetscCall(VecZeroEntries(globalFaceVec));
    PetscCall(DMLocalToGlobal(faceDm, localFaceVec, ADD_VALUES, globalFaceVec));

    // write to the output file
    PetscCall(VecView(globalFaceVec, viewer));
    PetscCall(DMRestoreGlobalVector(faceDm, &globalFaceVec));

    // cleanup
    PetscCall(DMRestoreLocalVector(faceDm, &localFaceVec));
    PetscCall(DMRestoreLocalVector(GetSolver()->GetSubDomain().GetDM(), &locXVec));
    PetscCall(DMRestoreLocalVector(boundaryDm, &localBoundaryVec));
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::monitors::Monitor, ablate::monitors::BoundarySolverMonitor, "Outputs any provided information from the boundary time to the serializer.");
