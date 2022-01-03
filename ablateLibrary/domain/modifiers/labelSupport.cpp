#include "labelSupport.hpp"
#include "utilities/petscError.hpp"

void ablate::domain::modifiers::LabelSupport::DistributeLabel(DM dm, DMLabel label) {
    PetscSF sfPoint;
    PetscInt nroots;
    DMGetPointSF(dm, &sfPoint) >> checkError;
    PetscSFGetGraph(sfPoint, &nroots, NULL, NULL, NULL) >> checkError;
    if (nroots >= 0) {
        DMLabel lblRoots, lblLeaves;
        IS valueIS, pointIS;
        const PetscInt *values;
        PetscInt numValues, v;

        /* Pull point contributions from remote leaves into local roots */
        DMLabelGather(label, sfPoint, &lblLeaves) >> checkError;
        DMLabelGetValueIS(lblLeaves, &valueIS) >> checkError;
        ISGetLocalSize(valueIS, &numValues) >> checkError;
        ISGetIndices(valueIS, &values) >> checkError;
        for (v = 0; v < numValues; ++v) {
            const PetscInt value = values[v];

            DMLabelGetStratumIS(lblLeaves, value, &pointIS) >> checkError;
            DMLabelInsertIS(label, pointIS, value) >> checkError;
            ISDestroy(&pointIS) >> checkError;
        }
        ISRestoreIndices(valueIS, &values) >> checkError;
        ISDestroy(&valueIS) >> checkError;
        DMLabelDestroy(&lblLeaves) >> checkError;
        /* Push point contributions from roots into remote leaves */
        DMLabelDistribute(label, sfPoint, &lblRoots) >> checkError;
        DMLabelGetValueIS(lblRoots, &valueIS) >> checkError;
        ISGetLocalSize(valueIS, &numValues) >> checkError;
        ISGetIndices(valueIS, &values) >> checkError;
        for (v = 0; v < numValues; ++v) {
            const PetscInt value = values[v];

            DMLabelGetStratumIS(lblRoots, value, &pointIS) >> checkError;
            DMLabelInsertIS(label, pointIS, value) >> checkError;
            ISDestroy(&pointIS) >> checkError;
        }
        ISRestoreIndices(valueIS, &values) >> checkError;
        ISDestroy(&valueIS) >> checkError;
        DMLabelDestroy(&lblRoots) >> checkError;
    }
}
