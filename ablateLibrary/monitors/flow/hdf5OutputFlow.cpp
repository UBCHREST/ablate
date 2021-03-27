#include "hdf5OutputFlow.hpp"
#include <petsc.h>
#include <petscviewerhdf5.h>
#include "environment/runEnvironment.hpp"
#include "parser/registrar.hpp"
#include "utilities/petscError.hpp"

void ablate::monitors::flow::Hdf5OutputFlow::Register(std::shared_ptr<ablate::flow::Flow> flowIn) {
    // store the flow
    flow = flowIn;

    // build the file name
    outputFilePath = environment::RunEnvironment::Get().GetOutputDirectory() / (flow->GetName() + extension);

    // setup the petsc viewer
    PetscViewerHDF5Open(PETSC_COMM_WORLD, outputFilePath.string().c_str(), FILE_MODE_WRITE, &petscViewer) >> checkError;

    // Print the initial mesh
    DMView(flow->GetMesh().GetDomain(), petscViewer) >> checkError;

    if (flow->GetFlowData()->auxDm) {
        DMSetOutputSequenceNumber(flow->GetFlowData()->auxDm, 0, 0) >> checkError;
    }
}

PetscErrorCode ablate::monitors::flow::Hdf5OutputFlow::OutputFlow(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx) {
    PetscFunctionBeginUser;
    auto monitor = (ablate::monitors::flow::Hdf5OutputFlow *)mctx;
    auto flowData = monitor->flow->GetFlowData();
    VecView(flowData->flowField, monitor->petscViewer) >> checkError;

    if (flowData->auxField) {
        // copy over the sequence data from the main dm
        PetscReal dmTime;
        PetscInt dmSequence;
        DMGetOutputSequenceNumber(flowData->dm, &dmSequence, &dmTime) >> checkError;
        DMSetOutputSequenceNumber(flowData->auxDm, dmSequence, dmTime) >> checkError;

        Vec auxGlobalField;
        DMGetGlobalVector(flowData->auxDm, &auxGlobalField) >> checkError;

        // copy over the name of the auxFieldVector
        const char *name;
        PetscObjectGetName((PetscObject)flowData->auxField, &name) >> checkError;
        PetscObjectSetName((PetscObject)auxGlobalField, name) >> checkError;

        DMLocalToGlobal(flowData->auxDm, flowData->auxField, INSERT_VALUES, auxGlobalField) >> checkError;
        VecView(auxGlobalField, monitor->petscViewer) >> checkError;
        DMRestoreGlobalVector(flowData->auxDm, &auxGlobalField) >> checkError;
    }

    PetscFunctionReturn(0);
}

REGISTER_WITHOUT_ARGUMENTS(ablate::monitors::flow::FlowMonitor, ablate::monitors::flow::Hdf5OutputFlow, "outputs the flow mesh and solution vector to a hdf5 file");
