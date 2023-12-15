#include "printDomainSummary.hpp"
#include "monitors/logs/stdOut.hpp"
#include "utilities/petscUtilities.hpp"
void ablate::domain::modifiers::PrintDomainSummary::Modify(DM &dm) {
    // print the dm summary
    PetscViewer viewer;
    PetscViewerCreate(PetscObjectComm((PetscObject)dm), &viewer) >> utilities::PetscUtilities::checkError;
    PetscViewerSetType(viewer, PETSCVIEWERASCII) >> utilities::PetscUtilities::checkError;

    DMView(dm, viewer) >> utilities::PetscUtilities::checkError;

    PetscOptionsRestoreViewer(&viewer) >> utilities::PetscUtilities::checkError;

    // Print any additional information specific to ablate domain
    ablate::monitors::logs::StdOut log;
    log.Initialize(PetscObjectComm((PetscObject)dm));

    // Print the information about the mesh size
    PetscReal minCellRadius = 0.0;
    DMPlexGetGeometryFVM(dm, NULL, NULL, &minCellRadius) >> utilities::PetscUtilities::checkError;
    log.Printf("Ablate Information:\n");
    log.Printf("\tminCellRadius: %g m\n", (double)minCellRadius);
}

#include "registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::PrintDomainSummary, "Prints a short summary of the domain to std out");