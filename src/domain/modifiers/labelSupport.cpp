#include "labelSupport.hpp"
#include "utilities/petscUtilities.hpp"

void ablate::domain::modifiers::LabelSupport::DistributeLabel(DM dm, DMLabel label) {
    PetscSF sfPoint;
    PetscInt nroots;
    DMGetPointSF(dm, &sfPoint) >> utilities::PetscUtilities::checkError;
    PetscSFGetGraph(sfPoint, &nroots, NULL, NULL, NULL) >> utilities::PetscUtilities::checkError;
    if (nroots >= 0) {
        DMLabel lblRoots, lblLeaves;
        IS valueIS, pointIS;
        const PetscInt *values;
        PetscInt numValues, v;

        /* Pull point contributions from remote leaves into local roots */
        DMLabelGather(label, sfPoint, &lblLeaves) >> utilities::PetscUtilities::checkError;
        DMLabelGetValueIS(lblLeaves, &valueIS) >> utilities::PetscUtilities::checkError;
        ISGetLocalSize(valueIS, &numValues) >> utilities::PetscUtilities::checkError;
        ISGetIndices(valueIS, &values) >> utilities::PetscUtilities::checkError;
        for (v = 0; v < numValues; ++v) {
            const PetscInt value = values[v];

            DMLabelGetStratumIS(lblLeaves, value, &pointIS) >> utilities::PetscUtilities::checkError;
            DMLabelInsertIS(label, pointIS, value) >> utilities::PetscUtilities::checkError;
            ISDestroy(&pointIS) >> utilities::PetscUtilities::checkError;
        }
        ISRestoreIndices(valueIS, &values) >> utilities::PetscUtilities::checkError;
        ISDestroy(&valueIS) >> utilities::PetscUtilities::checkError;
        DMLabelDestroy(&lblLeaves) >> utilities::PetscUtilities::checkError;
        /* Push point contributions from roots into remote leaves */
        DMLabelDistribute(label, sfPoint, &lblRoots) >> utilities::PetscUtilities::checkError;
        DMLabelGetValueIS(lblRoots, &valueIS) >> utilities::PetscUtilities::checkError;
        ISGetLocalSize(valueIS, &numValues) >> utilities::PetscUtilities::checkError;
        ISGetIndices(valueIS, &values) >> utilities::PetscUtilities::checkError;
        for (v = 0; v < numValues; ++v) {
            const PetscInt value = values[v];

            DMLabelGetStratumIS(lblRoots, value, &pointIS) >> utilities::PetscUtilities::checkError;
            DMLabelInsertIS(label, pointIS, value) >> utilities::PetscUtilities::checkError;
            ISDestroy(&pointIS) >> utilities::PetscUtilities::checkError;
        }
        ISRestoreIndices(valueIS, &values) >> utilities::PetscUtilities::checkError;
        ISDestroy(&valueIS) >> utilities::PetscUtilities::checkError;
        DMLabelDestroy(&lblRoots) >> utilities::PetscUtilities::checkError;
    }
}
