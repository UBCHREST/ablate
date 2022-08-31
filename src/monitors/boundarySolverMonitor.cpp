#include "boundarySolverMonitor.hpp"
#include "io/interval/fixedInterval.hpp"

ablate::monitors::BoundarySolverMonitor::~BoundarySolverMonitor() {
    if (boundaryDm) {
        DMDestroy(&boundaryDm) >> checkError;
    }

    if (faceDm) {
        DMDestroy(&faceDm) >> checkError;
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
    DMGetCoordinateDM(solver->GetSubDomain().GetDM(), &coordDM) >> checkError;
    DMClone(solver->GetSubDomain().GetDM(), &boundaryDm) >> checkError;
    DMSetCoordinateDM(boundaryDm, coordDM) >> checkError;

    // Create a label in the dm copy to mark boundary faces
    DMCreateLabel(boundaryDm, "boundaryFaceLabel") >> checkError;
    DMLabel boundaryFaceLabel;
    DMGetLabel(boundaryDm, "boundaryFaceLabel", &boundaryFaceLabel) >> checkError;

    // Also create a section on each of the faces.  This needs to be a custom section
    PetscSection boundaryFaceSection;
    PetscSectionCreate(PetscObjectComm((PetscObject)boundaryDm), &boundaryFaceSection) >> checkError;
    // Set the max/min bounds
    PetscInt fStart, fEnd;
    DMPlexGetHeightStratum(solver->GetSubDomain().GetDM(), 1, &fStart, &fEnd) >> checkError;
    PetscSectionSetChart(boundaryFaceSection, fStart, fEnd) >> checkError;

    // default section dof to zero
    for (PetscInt f = fStart; f < fEnd; ++f) {
        PetscSectionSetDof(boundaryFaceSection, f, 0) >> checkError;
    }

    // set the label at each of the faces and set the dof at each point
    const auto numberOfComponents = (PetscInt)boundarySolver->GetOutputComponents().size();
    for (const auto& gradientStencil : boundarySolver->GetBoundaryGeometry()) {
        // set both the label (used for filtering) and section for global variable creation
        DMLabelSetValue(boundaryFaceLabel, gradientStencil.geometry.faceId, 1) >> checkError;

        // set the dof at each section to the numberOfComponents
        PetscSectionSetDof(boundaryFaceSection, gradientStencil.geometry.faceId, numberOfComponents) >> checkError;
    }

    // finish the section
    PetscSectionSetUp(boundaryFaceSection) >> checkError;
    DMSetLocalSection(boundaryDm, boundaryFaceSection) >> checkError;
    PetscSectionDestroy(&boundaryFaceSection) >> checkError;

    // Complete the label
    DMPlexLabelComplete(boundaryDm, boundaryFaceLabel) >> checkError;

    // Now create a sub dm with only the faces
    DMPlexFilter(boundaryDm, boundaryFaceLabel, 1, &faceDm) >> checkError;

    // Add each of the output components on each face in the faceDm
    for (const auto& field : boundarySolver->GetOutputComponents()) {
        PetscFV fvm;
        PetscFVCreate(PetscObjectComm(PetscObject(faceDm)), &fvm) >> checkError;
        PetscObjectSetName((PetscObject)fvm, field.c_str()) >> checkError;
        PetscFVSetFromOptions(fvm) >> checkError;
        PetscFVSetNumComponents(fvm, 1) >> checkError;
        PetscInt dim;
        DMGetCoordinateDim(faceDm, &dim) >> checkError;
        PetscFVSetSpatialDimension(fvm, dim) >> checkError;

        DMAddField(faceDm, nullptr, (PetscObject)fvm) >> checkError;
        PetscFVDestroy(&fvm);
    }
    DMCreateDS(faceDm) >> checkError;
}

void ablate::monitors::BoundarySolverMonitor::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    // If this is the first output, store a copy of the faceDm
    if (sequenceNumber == 0) {
        DMView(faceDm, viewer) >> checkError;
    }

    // Set the output sequence number to the face dm
    DMSetOutputSequenceNumber(faceDm, sequenceNumber, time) >> checkError;

    // Create a local version of the solution (X) vector
    Vec locXVec;
    DMGetLocalVector(GetSolver()->GetSubDomain().GetDM(), &locXVec) >> checkError;
    DMGlobalToLocalBegin(GetSolver()->GetSubDomain().GetDM(), GetSolver()->GetSubDomain().GetSolutionVector(), INSERT_VALUES, locXVec) >> checkError;

    // create a local vector on the boundary solver
    Vec localBoundaryVec;
    DMGetLocalVector(boundaryDm, &localBoundaryVec) >> checkError;

    // finish with the locXVec
    DMGlobalToLocalEnd(GetSolver()->GetSubDomain().GetDM(), GetSolver()->GetSubDomain().GetSolutionVector(), INSERT_VALUES, locXVec) >> checkError;

    // compute the rhs
    boundarySolver->ComputeRHSFunction(time, locXVec, localBoundaryVec, boundarySolver->GetOutputFunctions()) >> checkError;

    // Create a local vector for just the monitor
    // create a local vector on the boundary solver
    Vec globalFaceVec;
    DMGetGlobalVector(faceDm, &globalFaceVec) >> checkError;
    PetscObjectSetName((PetscObject)globalFaceVec, GetId().c_str()) >> checkError;

    // map to the local face ids in the faceDm
    IS subPointIs;
    DMPlexGetSubpointIS(faceDm, &subPointIs) >> checkError;
    PetscInt size;
    VecGetSize(localBoundaryVec, &size);
    VecGetSize(globalFaceVec, &size);
    ISGetSize(subPointIs, &size);

    VecISCopy(localBoundaryVec, subPointIs, SCATTER_REVERSE, globalFaceVec) >> checkError;

    // write to the output file
    VecView(globalFaceVec, viewer) >> checkError;

    // cleanup
    DMRestoreGlobalVector(faceDm, &globalFaceVec) >> checkError;
    DMRestoreLocalVector(GetSolver()->GetSubDomain().GetDM(), &locXVec) >> checkError;
    DMRestoreLocalVector(boundaryDm, &localBoundaryVec) >> checkError;
}

#include "registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::monitors::Monitor, ablate::monitors::BoundarySolverMonitor, "Outputs any provided information from the boundary time to the serializer.");
