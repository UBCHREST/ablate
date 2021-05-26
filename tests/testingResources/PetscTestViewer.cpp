#include "PetscTestViewer.hpp"
#include "petscviewer.h"

testingResources::PetscTestViewer::PetscTestViewer(MPI_Comm comm) {
    // Create a temporary file
    file = std::tmpfile();

    // Use this to create an ascii viewer
    PetscErrorCode ierr = PetscViewerASCIIOpenWithFILE(comm, file, &viewer);
    if (ierr > 0) {
        throw std::runtime_error("cannot create PetscTestViewer");
    }
}

testingResources::PetscTestViewer::~PetscTestViewer() {
    // close the file
    std::fclose(file);
    file = NULL;

    // and destroy the viewer
    PetscViewerDestroy(&viewer);
}
std::string testingResources::PetscTestViewer::GetString() {
    // build a string stream to hold the file
    std::stringstream stream;

    // move the file to the start
    std::rewind(file);

    char buffer[128];
    while (std::fgets(buffer, sizeof(buffer), file)) {
        stream << buffer;
    }

    return stream.str();
}
