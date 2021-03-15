#include "hdf5OutputFlow.hpp"
#include <petsc.h>
#include <petscviewerhdf5.h>
#include "generators.hpp"
#include "monitors/runEnvironment.hpp"
#include "parser/registrar.hpp"
#include "utilities/petscError.hpp"

void ablate::monitors::flow::Hdf5OutputFlow::Register(std::shared_ptr<ablate::flow::Flow> flowIn) {
    // store the flow
    flow = flowIn;

    // build the file name
    outputFilePath = monitors::RunEnvironment::Get().GetOutputDirectory() / (flow->GetName() + extension);

    // setup the petsc viewer
    PetscViewerHDF5Open(PETSC_COMM_WORLD, outputFilePath.string().c_str(), FILE_MODE_WRITE, &petscViewer) >> checkError;

    // Print the initial mesh
    DMView(flow->GetMesh().GetDomain(), petscViewer) >> checkError;
}

PetscErrorCode ablate::monitors::flow::Hdf5OutputFlow::OutputFlow(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx) {
    PetscFunctionBeginUser;
    auto monitor = (ablate::monitors::flow::Hdf5OutputFlow*) mctx;
    VecView(monitor->flow->GetSolutionVector(), monitor->petscViewer) >> checkError;
    PetscFunctionReturn(0);
}

REGISTER_WITHOUT_ARGUMENTS(ablate::monitors::flow::FlowMonitor, ablate::monitors::flow::Hdf5OutputFlow, "outputs the flow mesh and solution vector to a hdf5 file");
