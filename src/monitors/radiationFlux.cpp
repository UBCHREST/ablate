#include "radiationFlux.hpp"

ablate::monitors::RadiationFlux::RadiationFlux(std::vector<std::shared_ptr<radiation::Radiation>> radiationIn, std::shared_ptr<domain::Region> radiationRegionIn)
    : radiation(std::move(radiationIn)), radiationRegion(std::move(radiationRegionIn)) {}

ablate::monitors::RadiationFlux::~RadiationFlux() {}

void ablate::monitors::RadiationFlux::Register(std::shared_ptr<solver::Solver> solver) {
    Monitor::Register(solver);

    //    boundarySolver = std::dynamic_pointer_cast<ablate::boundarySolver::BoundarySolver>(solver);
    //    if (!boundarySolver) {
    //        throw std::invalid_argument("The BoundarySolverMonitor monitor can only be used with ablate::boundarySolver::BoundarySolver");
    //    }

    /**
     * Initialize the ray tracers in the list that was provided to the monitor.
     * The ray tracing solvers will independently solve for the different radiation properties
     * models that were assigned to them so that the different radiation properties results can be compared to
     * one another.
     */
    DMLabel ghostLabel;
    DMGetLabel(solver->GetSubDomain().GetDM(), "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;
    DMLabel radiationRegionLabel;
    DMGetLabel(solver->GetSubDomain().GetDM(), radiationRegion->GetName().c_str(), &radiationRegionLabel) >> utilities::PetscUtilities::checkError;

    /** Get the face range of the boundary cells to initialize the rays with this range. Add all of the faces to this range that belong to the boundary solver.
     * The purpose of using a dynamic range is to avoid including the boundary cells within the stored range of faces that belongs to the radiation solvers in the monitor.
     * */
    // TODO: This is a kind of brute force method of getting the cells in the region that could be replaced later. It probably has minimal impact on performance as an initialization routine.
    ablate::solver::Range solverRange;
    solver->GetCellRange(solverRange);
    for (PetscInt c = solverRange.start; c < solverRange.end; ++c) {
        const PetscInt iCell = solverRange.points ? solverRange.points[c] : c;  //!< Isolates the valid cells
        PetscInt ghost = -1;
        PetscInt rad = -1;
        if (ghostLabel) DMLabelGetValue(ghostLabel, iCell, &ghost) >> utilities::PetscUtilities::checkError;
        if (radiationRegionLabel) DMLabelGetValue(radiationRegionLabel, iCell, &rad) >> utilities::PetscUtilities::checkError;
        if (!(ghost >= 0) && !(rad >= 0)) faceRange.Add(iCell);  //!< Add each ID to the range that the radiation solver will use
    }
    solver->RestoreRange(solverRange);
    // TODO: In the future just get the depth and do a two-iteration for loop through the depth and depth - 1
    solver->GetFaceRange(solverRange);
    for (PetscInt c = solverRange.start; c < solverRange.end; ++c) {
        const PetscInt iCell = solverRange.points ? solverRange.points[c] : c;  //!< Isolates the valid cells
        PetscInt ghost = -1;
        PetscInt rad = -1;
        if (ghostLabel) DMLabelGetValue(ghostLabel, iCell, &ghost) >> utilities::PetscUtilities::checkError;
        if (radiationRegionLabel) DMLabelGetValue(radiationRegionLabel, iCell, &rad) >> utilities::PetscUtilities::checkError;
        if (!(ghost >= 0) && !(rad >= 0)) faceRange.Add(iCell);  //!< Add each ID to the range that the radiation solver will use
    }
    solver->RestoreRange(solverRange);

    for (auto& rayTracer : radiation) {
        rayTracer.Setup(faceRange.GetRange(), solver->GetSubDomain());
        rayTracer.Initialize(faceRange.GetRange(), solver->GetSubDomain());
    }
}

PetscErrorCode ablate::monitors::RadiationFlux::MonitorRadiation(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx) {
    PetscFunctionBeginUser;

    // TODO: The TCP monitor must store one output for each of the radiation models that are in the vector of ray tracing solvers.
    // The ratio of red to green intensities must be computed and output as well.
    // It is not clear whether the red to green intensity ratio output should be implicitly defined in the input definition or whether there should be an explicit definition of which absorption model
    // represents the red and green intensities respectively. THe definition of a helper class which represents two radiation solvers each carrying a red and green ray tracing solver would likely be
    // beneficial for the definition of the models.

    auto monitor = (ablate::monitors::RadiationFlux*)ctx;

    /**
     * First solve the radiation through each of the ray tracing solvers
     */
    for (auto& rayTracer : monitor->radiation) {
        rayTracer->EvaluateGains(
            monitor->GetSolver()->GetSubDomain().GetSolutionVector(), monitor->GetSolver()->GetSubDomain().GetField("temperature"), monitor->GetSolver()->GetSubDomain().GetAuxVector());
    }

    /**
     * After the radiation solution is computed, then the intensity of the individual radiation solutions can be output for each face.
     */
    auto& range = monitor->faceRange.GetRange();
    for (auto& rayTracer : monitor->radiation) {
        for (PetscInt c = range.start; c < range.end; ++c) {
//            const PetscInt iCell = range.GetPoint(c);  //!< Isolates the valid cells
            rayTracer->GetIntensity(0, monitor->faceRange.GetRange(), 0, 1);
            // TODO: This may need a different implementation in the orthogonal radiation so that the spherical geometry is not accidentally captured in the intensity.

            /**
             * Now that the intensity has been read out of the ray tracing solver, it will need to be written to the field which stores the radiation information in the monitor.
             */
            // TODO: Is the radiation information output to its own DM or is there another special place that the information should go?
        }
    }

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::RadiationFlux, "Outputs radiation flux information about a region.",
         ARG(std::vector<ablate::radiation::Radiation>, "radiation", "ray tracing solvers which write information to the boundary faces. Use orthogonal for a window or surface for a plate."),
         ARG(ablate::domain::Region, "region", "region where the radiation is detected."));
