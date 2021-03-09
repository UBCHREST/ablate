#include "hdf5Output.hpp"
#include "utilities/petscError.hpp"
#include "monitors/runEnvironment.hpp"
#include <petscviewerhdf5.h>
#include <petsc.h>
#include "parser/registrar.hpp"
void ablate::monitors::flow::Hdf5Output::Register(std::shared_ptr<ablate::flow::Flow> flowIn) {
    // store the flow
    flow = flowIn;

    // build the file name
    auto fileName = monitors::RunEnvironment::Get().GetOutputDirectory() / (flow->GetName() + extension);

    // setup the petsc viewer
    PetscViewerHDF5Open(PETSC_COMM_WORLD, fileName.string().c_str(), FILE_MODE_WRITE, &petscViewer) >> checkError;

    // Print the initial mesh
    DMView(flow->GetMesh().GetDomain(), petscViewer) >> checkError;
}

PetscErrorCode ablate::monitors::flow::Hdf5Output::OutputFlow(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx) {
    PetscFunctionBeginUser;
    auto monitor = (ablate::monitors::flow::Hdf5Output*) mctx;
    VecView(monitor->flow->GetSolutionVector(), monitor->petscViewer) >> checkError;
    PetscFunctionReturn(0);
}

ablate::monitors::flow::Hdf5Output::~Hdf5Output() {
    if(petscViewer){
        PetscViewerDestroy(&petscViewer) >> checkError;
    }
}

REGISTER_WITHOUT_ARGUMENTS(ablate::monitors::flow::Monitor, ablate::monitors::flow::Hdf5Output, "outputs the flow mesh and solution vector to a hdf5 file");
