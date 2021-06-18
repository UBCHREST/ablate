#include "fvFlow.hpp"
#include <utilities/petscError.hpp>
ablate::flow::FVFlow::FVFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<parameters::Parameters> options,
                             std::vector<std::shared_ptr<mathFunctions::FieldSolution>> initialization, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                             std::vector<std::shared_ptr<mathFunctions::FieldSolution>> auxiliaryFields, std::vector<std::shared_ptr<mathFunctions::FieldSolution>> exactSolution)
    : Flow(name, mesh, parameters, options, initialization, boundaryConditions, auxiliaryFields, exactSolution) {}

PetscErrorCode ablate::flow::FVFlow::FVRHSFunctionLocal(DM dm, PetscReal time, Vec locXVec, Vec globFVec, void *ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    ablate::flow::FVFlow* flow = (ablate::flow::FVFlow*)ctx;

    /* Handle non-essential (e.g. outflow) boundary values.  This should be done before the auxFields are updated so that boundary values can be updated */
    Vec facegeom, cellgeom;
    ierr = DMPlexGetGeometryFVM(dm, &facegeom, &cellgeom, NULL);CHKERRQ(ierr);
    ierr = DMPlexInsertBoundaryValues(dm, PETSC_FALSE, locXVec, time, facegeom, cellgeom, NULL);CHKERRQ(ierr);

    // update any aux fields, including ghost cells
    ierr = FVFlowUpdateAuxFieldsFV(flow->dm->GetDomain(), flow->auxDM, time, locXVec, flow->auxField, flow->auxFieldUpdateFunctions.size(), &flow->auxFieldUpdateFunctions[0], &flow->auxFieldUpdateContexts[0]);
    CHKERRQ(ierr);

    // compute the euler flux across each face (note CompressibleFlowComputeEulerFlux has already been registered)
    ierr = ABLATE_DMPlexTSComputeRHSFunctionFVM(&flow->rhsFunctionDescriptions[0], flow->rhsFunctionDescriptions.size(), dm, time, locXVec, globFVec);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}
void ablate::flow::FVFlow::CompleteProblemSetup(TS ts) {
    Flow::CompleteProblemSetup(ts);


    // Override the DMTSSetRHSFunctionLocal in DMPlexTSComputeRHSFunctionFVM with a function that includes euler and diffusion source terms
    DMTSSetRHSFunctionLocal(dm->GetDomain(), FVRHSFunctionLocal, this) >> checkError;

    // copy over any boundary information from the dm, to the aux dm and set the sideset
    if (auxDM) {
        PetscDS flowProblem;
        DMGetDS(dm->GetDomain(), &flowProblem) >> checkError;
        PetscDS auxProblem;
        DMGetDS(auxDM, &auxProblem) >> checkError;

        // Get the number of boundary conditions and other info
        PetscInt numberBC;
        PetscDSGetNumBoundary(flowProblem, &numberBC) >> checkError;
        PetscInt numberAuxFields;
        PetscDSGetNumFields(auxProblem, &numberAuxFields) >> checkError;

        for (PetscInt bc = 0; bc < numberBC; bc++) {
            DMBoundaryConditionType type;
            const char* name;
            const char* labelName;
            PetscInt field;
            PetscInt numberIds;
            const PetscInt* ids;

            // Get the boundary
            PetscDSGetBoundary(flowProblem, bc, &type, &name, &labelName, &field, NULL, NULL, NULL, NULL, &numberIds, &ids, NULL) >> checkError;

            // If this is for euler and DM_BC_NATURAL_RIEMANN add it to the aux
            if (type == DM_BC_NATURAL_RIEMANN && field == 0) {
                for (PetscInt af = 0; af < numberAuxFields; af++) {
                    PetscDSAddBoundary(auxProblem, type, name, labelName, af, 0, NULL, NULL, NULL, numberIds, ids, NULL) >> checkError;
                }
            }
        }
    }

}
