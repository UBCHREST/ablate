#include "hdf5Monitor.hpp"
#include <petscviewerhdf5.h>
#include <environment/runEnvironment.hpp>
#include "generators.hpp"
#include "utilities/petscError.hpp"

ablate::monitors::Hdf5Monitor::~Hdf5Monitor() {
    if (petscViewer) {
        PetscViewerDestroy(&petscViewer) >> checkError;
    }

    // If this is the root process generate the xdmf file
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0 && !outputFilePath.empty() && std::filesystem::exists(outputFilePath)) {
        petscXdmfGenerator::Generate(outputFilePath);
    }
}
void ablate::monitors::Hdf5Monitor::Register(std::shared_ptr<Monitorable> object) {
    // cast the object and check if it is viewable
    viewableObject = std::dynamic_pointer_cast<Viewable>(object);

    // build the file name
    outputFilePath = environment::RunEnvironment::Get().GetOutputDirectory() / (viewableObject->GetName() + extension);

    // setup the petsc viewer
    PetscViewerHDF5Open(PETSC_COMM_WORLD, outputFilePath.string().c_str(), FILE_MODE_WRITE, &petscViewer) >> checkError;
}

PetscErrorCode ablate::monitors::Hdf5Monitor::OutputHdf5(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx) {
    PetscFunctionBeginUser;
    auto monitor = (ablate::monitors::Hdf5Monitor *)mctx;
    auto monitorObject = monitor->viewableObject;

    if (steps == 0 || monitor->interval == 0 || (steps % monitor->interval == 0)) {
        try {
            monitorObject->View(monitor->petscViewer, monitor->index, time, u);
        } catch (std::exception &e) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, e.what());
        }
        monitor->index++;
    }
    PetscFunctionReturn(0);
}
ablate::monitors::Hdf5Monitor::Hdf5Monitor(int interval) : interval(interval) {}

#include "parser/registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::Hdf5Monitor, "writes the viewable object to an hdf5", ARG(int, "interval", "how often to write the HDF5 file (default is every timestep)"));
