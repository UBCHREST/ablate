#include "virtualTcp.hpp"

ablate::monitors::VirtualTcp::VirtualTcp(std::vector<ablate::radiation::Radiation> radiationIn) : radiation(std::move(radiationIn)) {}

ablate::monitors::VirtualTcp::~VirtualTcp() {}

void ablate::monitors::VirtualTcp::Register(std::shared_ptr<solver::Solver> solver) {
    Monitor::Register(solver);

    boundarySolver = std::dynamic_pointer_cast<ablate::boundarySolver::BoundarySolver>(solver);
    if (!boundarySolver) {
        throw std::invalid_argument("The BoundarySolverMonitor monitor can only be used with ablate::boundarySolver::BoundarySolver");
    }

    /**
     * Initialize the ray tracers in the list that was provided to the monitor.
     * The ray tracing solvers will independently solve for the different radiation properties
     * models that were assigned to them so that the different radiation properties results can be compared to
     * one another.
     */
    DMLabel ghostLabel;
    DMGetLabel(solver->GetSubDomain().GetDM(), "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

    /** Get the face range of the boundary cells to initialize the rays with this range. Add all of the faces to this range that belong to the boundary solver.
     * The purpose of using a dynamic range is to avoid including the boundary cells within the stored range of faces that belongs to the radiation solvers in the monitor.
     * */

    for (const auto& i : boundarySolver->GetBoundaryGeometry()) {
        PetscInt ghost = -1;
        if (ghostLabel) DMLabelGetValue(ghostLabel, i.geometry.faceId, &ghost) >> utilities::PetscUtilities::checkError;
        if (!(ghost >= 0)) faceRange.Add(i.geometry.faceId);  //!< Add each ID to the range that the radiation solver will use
    }

    for (auto& rayTracer : radiation) {
        rayTracer.Setup(faceRange.GetRange(), solver->GetSubDomain());
        rayTracer.Initialize(faceRange.GetRange(), solver->GetSubDomain());
    }
}

void ablate::monitors::VirtualTcp::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    PetscFunctionBeginUser;

    // TODO: The TCP monitor must store one output for each of the radiation models that are in the vector of ray tracing solvers.
    // The ratio of red to green intensities must be computed and output as well.
    // It is not clear whether the red to green intensity ratio output should be implicitly defined in the input definition or whether there should be an explicit definition of which absorption model
    // represents the red and green intensities respectively. THe definition of a helper class which represents two radiation solvers each carrying a red and green ray tracing solver would likely be
    // beneficial for the definition of the models.

    /**
     * First solve the radiation through each of the ray tracing solvers
     */
    for (auto& rayTracer : radiation) {
        rayTracer.EvaluateGains(GetSolver()->GetSubDomain().GetSolutionVector(), GetSolver()->GetSubDomain().GetField("temperature"), GetSolver()->GetSubDomain().GetAuxVector());
    }

    /**
     * After the radiation solution is computed, then the intensity of the individual radiation solutions can be output for each face.
     */
    auto& range = faceRange.GetRange();
    for (auto& rayTracer : radiation) {
        for (PetscInt c = range.start; c < range.end; ++c) {
            const PetscInt iCell = range.GetPoint(c);  //!< Isolates the valid cells
            rayTracer.GetIntensity(0, faceRange.GetRange(), 0, 1);
            // TODO: This may need a different implementation in the orthogonal radiation so that the spherical geometry is not accidentally captured in the intensity.

            /**
             * Now that the intensity has been read out of the ray tracing solver, it will need to be written to the field which stores the radiation information in the monitor.
             */


        }
    }

    PetscFunctionReturnVoid();
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::VirtualTcp, "Outputs TCP information to the serializer.",
         ARG(std::vector<ablate::radiation::Radiation>, "radiation", "ray tracing solvers which write information to the boundary faces."));
