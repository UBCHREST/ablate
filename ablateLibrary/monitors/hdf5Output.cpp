#include "hdf5Output.hpp"
#include <petscviewerhdf5.h>
#include "generators.hpp"
#include "utilities/petscError.hpp"

ablate::monitors::Hdf5Output::~Hdf5Output() {
    if (petscViewer) {
        PetscViewerDestroy(&petscViewer) >> checkError;
    }

    // If this is the root process generate the xdmf file
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0 && !outputFilePath.empty() && std::filesystem::exists(outputFilePath)) {
        petscXdmfGenerator::Generate(outputFilePath);
    }
}