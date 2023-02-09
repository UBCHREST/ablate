#include "radiationFlux.hpp"

ablate::monitors::RadiationFlux::RadiationFlux(std::vector<std::shared_ptr<radiation::Radiation>> radiationIn, std::shared_ptr<domain::Region> radiationFluxRegionIn)
    : radiation(std::move(radiationIn)), radiationFluxRegion(std::move(radiationFluxRegionIn)) {}

ablate::monitors::RadiationFlux::~RadiationFlux() {}

void ablate::monitors::RadiationFlux::Register(std::shared_ptr<solver::Solver> solverIn) {
    Monitor::Register(solverIn);

    // TODO: Put in a condition such that the radiation flux monitor can only take radiation classes with surface based implementations.

    // TODO: Run a check on all of the indexes in the given region to ensure that they are faces and not cells.

    //    GetSolver() = std::dynamic_pointer_cast<ablate::boundarySolver::BoundarySolver>(solverIn);
    //    if (!boundarySolver) {
    //        throw std::invalid_argument("The BoundarySolverMonitor monitor can only be used with ablate::boundarySolver::BoundarySolver");
    //    }

    // update the name
    name = radiationFluxRegion->GetName() + name;

    DMLabel radiationFluxRegionLabel = nullptr;
    PetscInt regionValue = 0;
    domain::Region::GetLabel(radiationFluxRegion, solverIn->GetSubDomain().GetDM(), radiationFluxRegionLabel, regionValue);

    // Now create a sub dm with only the faces
    DMPlexFilter(solverIn->GetSubDomain().GetDM(), radiationFluxRegionLabel, 1, &fluxDm) >> utilities::PetscUtilities::checkError;

    /** Add each of the output components on each face in the fluxDm
     * the number of components should be equal to the number of ray tracers plus any ratio outputs?
     */
    for (const auto& rayTracer : radiation) {
        PetscFV fvm;
        PetscFVCreate(PetscObjectComm(PetscObject(fluxDm)), &fvm) >> utilities::PetscUtilities::checkError;
        PetscObjectSetName((PetscObject)fvm, rayTracer->GetId().c_str()) >> utilities::PetscUtilities::checkError;
        PetscFVSetFromOptions(fvm) >> utilities::PetscUtilities::checkError;
        PetscFVSetNumComponents(fvm, 1) >> utilities::PetscUtilities::checkError;
        PetscInt dim;
        DMGetCoordinateDim(fluxDm, &dim) >> utilities::PetscUtilities::checkError;
        PetscFVSetSpatialDimension(fvm, dim) >> utilities::PetscUtilities::checkError;

        DMAddField(fluxDm, nullptr, (PetscObject)fvm) >> utilities::PetscUtilities::checkError;
        PetscFVDestroy(&fvm);
    }
    DMCreateDS(fluxDm) >> utilities::PetscUtilities::checkError;

    /**
     * Initialize the ray tracers in the list that was provided to the monitor.
     * The ray tracing solvers will independently solve for the different radiation properties
     * models that were assigned to them so that the different radiation properties results can be compared to
     * one another.
     */
    DMLabel ghostLabel;
    DMGetLabel(solverIn->GetSubDomain().GetDM(), "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;
    DMLabel radiationRegionLabel;
    DMGetLabel(solverIn->GetSubDomain().GetDM(), radiationFluxRegion->GetName().c_str(), &radiationRegionLabel) >> utilities::PetscUtilities::checkError;

    // March over each cell in the face dm
    PetscInt cStart, cEnd;
    DMPlexGetHeightStratum(fluxDm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;

    // get the mapping information
    IS faceIs;
    const PetscInt* faceToBoundary = nullptr;
    DMPlexGetSubpointIS(fluxDm, &faceIs) >> utilities::PetscUtilities::checkError;
    ISGetIndices(faceIs, &faceToBoundary) >> utilities::PetscUtilities::checkError;

    /** Get the face range of the boundary cells to initialize the rays with this range. Add all of the faces to this range that belong to the boundary solverIn.
     * The purpose of using a dynamic range is to avoid including the boundary cells within the stored range of faces that belongs to the radiation solvers in the monitor.
     * */
    //    ablate::solver::Range solverRange;
    //    solverIn->GetFaceRange(solverRange);
    for (PetscInt c = cStart; c < cEnd; ++c) {
        const PetscInt iCell = faceToBoundary[c];  //!< Isolates the valid cells
        PetscInt ghost = -1;
        if (ghostLabel) DMLabelGetValue(ghostLabel, iCell, &ghost) >> utilities::PetscUtilities::checkError;
        if (!(ghost >= 0) && (radiationFluxRegion->InRegion(radiationFluxRegion, GetSolver()->GetSubDomain().GetDM(), iCell)))
            monitorRange.Add(iCell);  //!< Add each ID to the range that the radiation solverIn will use
    }
    // restore
    ISRestoreIndices(faceIs, &faceToBoundary) >> utilities::PetscUtilities::checkError;
    //    solverIn->RestoreRange(solverRange);

    for (auto& rayTracer : radiation) {
        rayTracer->Setup(monitorRange.GetRange(), solverIn->GetSubDomain());
        rayTracer->Initialize(monitorRange.GetRange(), solverIn->GetSubDomain());
    }
}

void ablate::monitors::RadiationFlux::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    PetscFunctionBeginUser;

    // If this is the first output, store a copy of the fluxDm
    if (sequenceNumber == 0) {
        DMView(fluxDm, viewer) >> utilities::PetscUtilities::checkError;
    }

    // Set the output sequence number to the face dm
    DMSetOutputSequenceNumber(fluxDm, sequenceNumber, time) >> utilities::PetscUtilities::checkError;

    /** Get read access to the local solution information
     */
    const PetscScalar* locXArray;
    VecGetArrayRead(GetSolver()->GetSubDomain().GetSolutionVector(), &locXArray) >> utilities::PetscUtilities::checkError;

    // Create a local vector for just the monitor
    Vec localFaceVec;
    DMGetLocalVector(fluxDm, &localFaceVec) >> utilities::PetscUtilities::checkError;
    VecZeroEntries(localFaceVec) >> utilities::PetscUtilities::checkError;

    // Get the raw data for the global vectors
    PetscScalar* localFaceArray;
    VecGetArray(localFaceVec, &localFaceArray) >> utilities::PetscUtilities::checkError;

    // Determine the size of data
    PetscInt dataSize;
    VecGetBlockSize(localFaceVec, &dataSize) >> utilities::PetscUtilities::checkError;

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
    if (locXArray && localFaceArray) {

        // March over each cell in the face dm
        PetscInt cStart, cEnd;
        DMPlexGetHeightStratum(fluxDm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;

        // get the mapping information
        IS faceIs;
        const PetscInt* faceToBoundary = nullptr;
        DMPlexGetSubpointIS(fluxDm, &faceIs) >> utilities::PetscUtilities::checkError;
        ISGetIndices(faceIs, &faceToBoundary) >> utilities::PetscUtilities::checkError;

        for (int i = 0; i < int(radiation.size()); i++) {
            for (PetscInt c = cStart; c < cEnd; ++c) {
                const PetscInt iCell = faceToBoundary[c];  //!< Isolates the valid cells
                /**
                 * Write the intensity into the fluxDm for outputting.
                 * Now that the intensity has been read out of the ray tracing solver, it will need to be written to the field which stores the radiation information in the monitor.
                 * This is where the computed information should be written to the dm that was created for the radiation flux monitor.
                 */
                PetscScalar* globalFaceData = nullptr;  // TODO: Is the size of the pointer that is retrieved equal to the number of fields that are in the DM. Are they in order still?
                DMPlexPointLocalRef(fluxDm, c, localFaceArray, &globalFaceData) >> utilities::PetscUtilities::checkError;
                /**
                 * Get the intensity calculated out of the ray tracer. Write it to the appropriate location in the face DM.
                 */
                globalFaceData[i] = radiation[i]->GetIntensity(iCell, monitorRange.GetRange(), 0, 1);
            }
        }
        // restore
        ISRestoreIndices(faceIs, &faceToBoundary) >> utilities::PetscUtilities::checkError;
    }

    VecRestoreArray(localFaceVec, &localFaceArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(GetSolver()->GetSubDomain().GetSolutionVector(), &locXArray) >> utilities::PetscUtilities::checkError;

    // Map to a global array with add values
    Vec globalFaceVec;
    DMGetGlobalVector(fluxDm, &globalFaceVec) >> utilities::PetscUtilities::checkError;
    PetscObjectSetName((PetscObject)globalFaceVec, GetId().c_str()) >> utilities::PetscUtilities::checkError;
    VecZeroEntries(globalFaceVec);
    DMLocalToGlobal(fluxDm, localFaceVec, ADD_VALUES, globalFaceVec) >> utilities::PetscUtilities::checkError;

    // write to the output file
    VecView(globalFaceVec, viewer) >> utilities::PetscUtilities::checkError;
    DMRestoreGlobalVector(fluxDm, &globalFaceVec) >> utilities::PetscUtilities::checkError;

    PetscFunctionReturnVoid();
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::RadiationFlux, "Outputs radiation flux information about a region.",
         ARG(std::vector<ablate::radiation::Radiation>, "radiation", "ray tracing solvers which write information to the boundary faces. Use orthogonal for a window or surface for a plate."),
         ARG(ablate::domain::Region, "region", "region where the radiation is detected."));
