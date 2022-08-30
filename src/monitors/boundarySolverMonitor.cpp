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

    // Create a dm that holds a single set of values at each face in this label
    PetscInt fStart, fEnd;
    DMPlexGetHeightStratum(solver->GetSubDomain().GetDM(), 1, &fStart, &fEnd) >> checkError;

    // create a face solution dm for that is the required number of variables per face
    PetscSection solutionSection;
    PetscSectionCreate(PetscObjectComm((PetscObject)boundaryDm), &solutionSection) >> checkError;
    PetscSectionSetChart(solutionSection, fStart, fEnd) >> checkError;


    // default section dof to zero
    for (PetscInt f = fStart; f < fEnd; ++f) {
        PetscSectionSetDof(solutionSection, f, 0) >> checkError;
    }

    // Set the rest to number of fields
    const auto numberOfComponents = (PetscInt)boundarySolver->GetOutputComponents().size();
    for (const auto& gradientStencil : boundarySolver->GetBoundaryGeometry()) {
            // set both the label (used for filtering) and section for global variable creation
            DMLabelSetValue(boundaryFaceLabel, gradientStencil.geometry.faceId, 1) >> checkError;
            PetscSectionSetDof(solutionSection, gradientStencil.geometry.faceId, numberOfComponents) >> checkError;
    }

    PetscSectionSetUp(solutionSection) >> checkError;
    DMSetLocalSection(boundaryDm, solutionSection) >> checkError;
    PetscSectionDestroy(&solutionSection) >> checkError;

    // Complete the label
    DMPlexLabelComplete(boundaryDm, boundaryFaceLabel) >> checkError;

    // Now create a sub dm with only the faces
//    DMPlexFilter(boundaryDm, boundaryFaceLabel, 1, &faceDm) >> checkError;
    DMPlexCreateSubmesh(boundaryDm, boundaryFaceLabel, 1, PETSC_TRUE, &faceDm)  >> checkError;
    DMView(faceDm, PETSC_VIEWER_STDOUT_WORLD);
    // create a helper function to add fields
    auto addField = [](DM& dm, const char* nameField, DMLabel label) {
        PetscFV fvm;
        PetscFVCreate(PetscObjectComm(PetscObject(dm)), &fvm) >> checkError;
        PetscObjectSetName((PetscObject)fvm, nameField) >> checkError;
        PetscFVSetFromOptions(fvm) >> checkError;
        PetscFVSetNumComponents(fvm, 1) >> checkError;

        DMAddField(dm, label, (PetscObject)fvm) >> checkError;
        PetscFVDestroy(&fvm);
    };
    for (const auto& component : boundarySolver->GetOutputComponents()) {
        addField(faceDm, component.c_str(), nullptr);
    }
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
