#include "radiationFlux.hpp"

ablate::monitors::RadiationFlux::RadiationFlux(std::vector<std::shared_ptr<radiation::Radiation>> radiationIn, std::shared_ptr<domain::Region> radiationFluxRegionIn)
    : radiation(std::move(radiationIn)), radiationFluxRegion(std::move(radiationFluxRegionIn)) {}

ablate::monitors::RadiationFlux::~RadiationFlux() {}

void ablate::monitors::RadiationFlux::Register(std::shared_ptr<solver::Solver> solver) {
    Monitor::Register(solver);

    // update the name
    name = radiationFluxRegion->GetName() + name;

    // make a copy of the dm for a boundary dm.
    DM coordDM;
    DMGetCoordinateDM(solver->GetSubDomain().GetDM(), &coordDM) >> utilities::PetscUtilities::checkError;
    DMClone(solver->GetSubDomain().GetDM(), &monitorDm) >> utilities::PetscUtilities::checkError;
    DMSetCoordinateDM(monitorDm, coordDM) >> utilities::PetscUtilities::checkError;

    // Create a label in the dm copy to mark boundary faces
    DMCreateLabel(monitorDm, "radiationFluxRegion") >> utilities::PetscUtilities::checkError;
    DMLabel radiationFluxLabel;
    DMGetLabel(monitorDm, "radiationFluxRegion", &radiationFluxLabel) >> utilities::PetscUtilities::checkError;

    // Also create a section on each of the faces.  This needs to be a custom section
    PetscSection boundaryFaceSection;
    PetscSectionCreate(PetscObjectComm((PetscObject)monitorDm), &boundaryFaceSection) >> utilities::PetscUtilities::checkError;
    // Set the max/min bounds
    PetscInt fStart, fEnd;
    DMPlexGetHeightStratum(solver->GetSubDomain().GetDM(), 1, &fStart, &fEnd) >> utilities::PetscUtilities::checkError;
    PetscSectionSetChart(boundaryFaceSection, fStart, fEnd) >> utilities::PetscUtilities::checkError;

    // default section dof to zero
    for (PetscInt f = fStart; f < fEnd; ++f) {
        PetscSectionSetDof(boundaryFaceSection, f, 0) >> utilities::PetscUtilities::checkError;
    }

    // set the label at each of the faces and set the dof at each point
    const auto numberOfComponents = (PetscInt)boundarySolver->GetOutputComponents().size();
    for (const auto& gradientStencil : boundarySolver->GetBoundaryGeometry()) {
        // set both the label (used for filtering) and section for global variable creation
        DMLabelSetValue(radiationFluxLabel, gradientStencil.geometry.faceId, 1) >> utilities::PetscUtilities::checkError;

        // set the dof at each section to the numberOfComponents
        PetscSectionSetDof(boundaryFaceSection, gradientStencil.geometry.faceId, numberOfComponents) >> utilities::PetscUtilities::checkError;
    }

    // finish the section
    PetscSectionSetUp(boundaryFaceSection) >> utilities::PetscUtilities::checkError;
    DMSetLocalSection(monitorDm, boundaryFaceSection) >> utilities::PetscUtilities::checkError;
    PetscSectionDestroy(&boundaryFaceSection) >> utilities::PetscUtilities::checkError;

    // Complete the label
    DMPlexLabelComplete(monitorDm, radiationFluxLabel) >> utilities::PetscUtilities::checkError;

    // Now create a sub dm with only the faces
    DMPlexFilter(monitorDm, radiationFluxLabel, 1, &faceDm) >> utilities::PetscUtilities::checkError;

    // Add each of the output components on each face in the faceDm
    for (const auto& field : boundarySolver->GetOutputComponents()) {
        PetscFV fvm;
        PetscFVCreate(PetscObjectComm(PetscObject(faceDm)), &fvm) >> utilities::PetscUtilities::checkError;
        PetscObjectSetName((PetscObject)fvm, field.c_str()) >> utilities::PetscUtilities::checkError;
        PetscFVSetFromOptions(fvm) >> utilities::PetscUtilities::checkError;
        PetscFVSetNumComponents(fvm, 1) >> utilities::PetscUtilities::checkError;
        PetscInt dim;
        DMGetCoordinateDim(faceDm, &dim) >> utilities::PetscUtilities::checkError;
        PetscFVSetSpatialDimension(fvm, dim) >> utilities::PetscUtilities::checkError;

        DMAddField(faceDm, nullptr, (PetscObject)fvm) >> utilities::PetscUtilities::checkError;
        PetscFVDestroy(&fvm);
    }
    DMCreateDS(faceDm) >> utilities::PetscUtilities::checkError;

    /**
     * Initialize the ray tracers in the list that was provided to the monitor.
     * The ray tracing solvers will independently solve for the different radiation properties
     * models that were assigned to them so that the different radiation properties results can be compared to
     * one another.
     */
    DMLabel ghostLabel;
    DMGetLabel(solver->GetSubDomain().GetDM(), "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;
    DMLabel radiationRegionLabel;
    DMGetLabel(solver->GetSubDomain().GetDM(), radiationFluxRegion->GetName().c_str(), &radiationRegionLabel) >> utilities::PetscUtilities::checkError;

    /** Get the face range of the boundary cells to initialize the rays with this range. Add all of the faces to this range that belong to the boundary solver.
     * The purpose of using a dynamic range is to avoid including the boundary cells within the stored range of faces that belongs to the radiation solvers in the monitor.
     * */
    ablate::solver::Range solverRange;
    solver->GetFaceRange(solverRange);
    for (PetscInt c = solverRange.start; c < solverRange.end; ++c) {
        const PetscInt iCell = solverRange.GetPoint(c);  //!< Isolates the valid cells
        PetscInt ghost = -1;
        PetscInt rad = -1;
        if (ghostLabel) DMLabelGetValue(ghostLabel, iCell, &ghost) >> utilities::PetscUtilities::checkError;
        if (radiationRegionLabel) DMLabelGetValue(radiationRegionLabel, iCell, &rad) >> utilities::PetscUtilities::checkError;
        if (!(ghost >= 0) && !(rad >= 0)) monitorRange.Add(iCell);  //!< Add each ID to the range that the radiation solver will use
    }
    solver->RestoreRange(solverRange);

    for (auto& rayTracer : radiation) {
        rayTracer->Setup(monitorRange.GetRange(), solver->GetSubDomain());
        rayTracer->Initialize(monitorRange.GetRange(), solver->GetSubDomain());
    }
}

void ablate::monitors::RadiationFlux::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    PetscFunctionBeginUser;

    // If this is the first output, store a copy of the faceDm
    if (sequenceNumber == 0) {
        DMView(faceDm, viewer) >> utilities::PetscUtilities::checkError;
    }

    // Set the output sequence number to the face dm
    DMSetOutputSequenceNumber(faceDm, sequenceNumber, time) >> utilities::PetscUtilities::checkError;

    // Create a local version of the solution (X) vector
    Vec locXVec;
    DMGetLocalVector(GetSolver()->GetSubDomain().GetDM(), &locXVec) >> utilities::PetscUtilities::checkError;
    DMGlobalToLocalBegin(GetSolver()->GetSubDomain().GetDM(), GetSolver()->GetSubDomain().GetSolutionVector(), INSERT_VALUES, locXVec) >> utilities::PetscUtilities::checkError;

    // create a local vector on the boundary solver
    Vec localBoundaryVec;
    DMGetLocalVector(GetSolver()->GetSubDomain().GetDM(), &localBoundaryVec) >> utilities::PetscUtilities::checkError;
    VecZeroEntries(localBoundaryVec) >> utilities::PetscUtilities::checkError;

    // finish with the locXVec
    DMGlobalToLocalEnd(GetSolver()->GetSubDomain().GetDM(), GetSolver()->GetSubDomain().GetSolutionVector(), INSERT_VALUES, locXVec) >> utilities::PetscUtilities::checkError;
/// We don't need to compute the rhs of the solver because we will do all of the radiation calculations
//    // compute the rhs
//    boundarySolver->ComputeRHSFunction(time, locXVec, localBoundaryVec, boundarySolver->GetOutputFunctions()) >> utilities::PetscUtilities::checkError;

    // Create a local vector for just the monitor
    Vec localFaceVec;
    DMGetLocalVector(faceDm, &localFaceVec) >> utilities::PetscUtilities::checkError;
    VecZeroEntries(localFaceVec) >> utilities::PetscUtilities::checkError;

    // Get the raw data for the global vectors
    const PetscScalar* localBoundaryArray;
    VecGetArrayRead(localBoundaryVec, &localBoundaryArray) >> utilities::PetscUtilities::checkError;
    PetscScalar* localFaceArray;
    VecGetArray(localFaceVec, &localFaceArray) >> utilities::PetscUtilities::checkError;

    // Determine the size of data
    PetscInt dataSize;
    VecGetBlockSize(localFaceVec, &dataSize) >> utilities::PetscUtilities::checkError;

    // March over each cell in the face dm
    PetscInt cStart, cEnd;
    DMPlexGetHeightStratum(faceDm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;

    // get the mapping information
    IS faceIs;
    const PetscInt* faceToBoundary = nullptr;
    DMPlexGetSubpointIS(faceDm, &faceIs) >> utilities::PetscUtilities::checkError;
    ISGetIndices(faceIs, &faceToBoundary) >> utilities::PetscUtilities::checkError;

    // TODO: The TCP monitor must store one output for each of the radiation models that are in the vector of ray tracing solvers.
    // The ratio of red to green intensities must be computed and output as well.
    // It is not clear whether the red to green intensity ratio output should be implicitly defined in the input definition or whether there should be an explicit definition of which absorption model
    // represents the red and green intensities respectively. THe definition of a helper class which represents two radiation solvers each carrying a red and green ray tracing solver would likely be
    // beneficial for the definition of the models.

    /**
     * First solve the radiation through each of the ray tracing solvers
     */
    for (auto& rayTracer : radiation) {
        rayTracer->EvaluateGains(GetSolver()->GetSubDomain().GetSolutionVector(), GetSolver()->GetSubDomain().GetField("temperature"), GetSolver()->GetSubDomain().GetAuxVector());
    }

    /**
     * After the radiation solution is computed, then the intensity of the individual radiation solutions can be output for each face.
     */
    if (localBoundaryArray && localFaceArray) {
        auto& range = monitorRange.GetRange();
        for (auto& rayTracer : radiation) {
            for (PetscInt c = range.start; c < range.end; ++c) {
                const PetscInt iCell = range.GetPoint(c);  //!< Isolates the valid cells
                rayTracer->GetIntensity(0, monitorRange.GetRange(), 0, 1);

                /**
                 * Now that the intensity has been read out of the ray tracing solver, it will need to be written to the field which stores the radiation information in the monitor.
                 */


                /// This is where the computed information should be written to the dm that was created for the radiation flux monitor.

                const PetscScalar* localBoundaryData = nullptr;
                PetscScalar* globalFaceData = nullptr;

                DMPlexPointLocalRead(monitorDm, iCell, localBoundaryArray, &localBoundaryData) >> utilities::PetscUtilities::checkError;
                DMPlexPointLocalRef(faceDm, c, localFaceArray, &globalFaceData) >> utilities::PetscUtilities::checkError; // TODO: Should c go here or something else?
                if (globalFaceData && localBoundaryData) {
                    PetscArraycpy(globalFaceData, localBoundaryData, dataSize) >> utilities::PetscUtilities::checkError;
                }
            }
        }
    }

    // restore
    ISRestoreIndices(faceIs, &faceToBoundary) >> utilities::PetscUtilities::checkError;

    VecRestoreArrayRead(localBoundaryVec, &localBoundaryArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArray(localFaceVec, &localFaceArray) >> utilities::PetscUtilities::checkError;

    // Map to a global array with add values
    Vec globalFaceVec;
    DMGetGlobalVector(faceDm, &globalFaceVec) >> utilities::PetscUtilities::checkError;
    PetscObjectSetName((PetscObject)globalFaceVec, GetId().c_str()) >> utilities::PetscUtilities::checkError;
    VecZeroEntries(globalFaceVec);
    DMLocalToGlobal(faceDm, localFaceVec, ADD_VALUES, globalFaceVec) >> utilities::PetscUtilities::checkError;

    // write to the output file
    VecView(globalFaceVec, viewer) >> utilities::PetscUtilities::checkError;
    DMRestoreGlobalVector(faceDm, &globalFaceVec) >> utilities::PetscUtilities::checkError;

    // cleanup
    DMRestoreLocalVector(faceDm, &localFaceVec) >> utilities::PetscUtilities::checkError;
    DMRestoreLocalVector(GetSolver()->GetSubDomain().GetDM(), &locXVec) >> utilities::PetscUtilities::checkError;
    DMRestoreLocalVector(monitorDm, &localBoundaryVec) >> utilities::PetscUtilities::checkError;

    PetscFunctionReturnVoid();
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::RadiationFlux, "Outputs radiation flux information about a region.",
         ARG(std::vector<ablate::radiation::Radiation>, "radiation", "ray tracing solvers which write information to the boundary faces. Use orthogonal for a window or surface for a plate."),
         ARG(ablate::domain::Region, "region", "region where the radiation is detected."));
