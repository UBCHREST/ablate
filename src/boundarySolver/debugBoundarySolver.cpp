#include "debugBoundarySolver.hpp"
#include <fstream>
#include <set>
#include "boundaryProcess.hpp"
#include "environment/runEnvironment.hpp"
#include "utilities/mpiUtilities.hpp"

ablate::boundarySolver::DebugBoundarySolver::DebugBoundarySolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<domain::Region> fieldBoundary,
                                                                 std::vector<std::shared_ptr<BoundaryProcess>> boundaryProcesses, std::shared_ptr<parameters::Parameters> options, bool mergeFaces)
    : BoundarySolver(solverId, region, fieldBoundary, boundaryProcesses, options, mergeFaces) {}

void ablate::boundarySolver::DebugBoundarySolver::Setup() {
    BoundarySolver::Setup();

    // Get the solver name
    const auto& solverName = GetSolverId();
    const auto& solverRegion = GetRegion();

    // check to make sure that gradientStencils is not zero
    PetscMPIInt numberStencilLocal = (PetscMPIInt)gradientStencils.size();
    PetscMPIInt numberStencilGlobal;
    MPI_Reduce(&numberStencilLocal, &numberStencilGlobal, 1, MPI_INT, MPI_SUM, 0, GetSubDomain().GetComm()) >> utilities::MpiUtilities::checkError;

    utilities::MpiUtilities::Once(
        GetSubDomain().GetComm(),
        [numberStencilGlobal, &solverName, &solverRegion]() {
            if (numberStencilGlobal == 0) {
                throw std::invalid_argument("The " + solverName + " setup resulted in zero boundary cells for region " + solverRegion->ToString());
            }
        },
        0);

    // Create an output directory for this solver
    auto solverDirectory = ablate::environment::RunEnvironment::Get().GetOutputDirectory() / GetSolverId();
    ablate::utilities::MpiUtilities::Once(GetSubDomain().GetComm(), [solverDirectory] { create_directories(solverDirectory); });

    // Determine the header
    std::string header = "p";
    switch (GetSubDomain().GetDimensions()) {
        case 1:
            header += " x";
            break;
        case 2:
            header += " x y";
            break;
        case 3:
            header += " x y z";
            break;
    }

    // Get the rank
    PetscMPIInt rank;
    MPI_Comm_rank(GetSubDomain().GetComm(), &rank) >> utilities::PetscUtilities::checkError;

    // march over each stencil and create a separate file
    for (std::size_t s = 0; s < gradientStencils.size(); ++s) {
        const auto& stencil = gradientStencils[s];

        std::ofstream stencilFile;
        stencilFile.open(solverDirectory / ("stencil_rank_" + std::to_string(rank) + "." + std::to_string(s) + ".txt"));

        // Output the header info
        stencilFile << header << std::endl;

        // Output the stencil location
        OutputStencilCellLocation(stencilFile, stencil.cellId);
        stencilFile << std::endl;

        // now output each stencil point
        for (PetscInt sp = 0; sp < stencil.stencilSize; sp++) {
            OutputStencilCellLocation(stencilFile, stencil.stencil[sp]);
            stencilFile << std::endl;
        }

        stencilFile.close();
    }
}

void ablate::boundarySolver::DebugBoundarySolver::OutputStencilCellLocation(std::ostream& stream, PetscInt cell) {
    stream << cell;

    const PetscScalar* cellGeomArray;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
    DM cellDM;
    VecGetDM(cellGeomVec, &cellDM) >> utilities::PetscUtilities::checkError;

    PetscFVCellGeom* cg;
    DMPlexPointLocalRead(cellDM, cell, cellGeomArray, &cg) >> utilities::PetscUtilities::checkError;

    for (PetscInt d = 0; d < GetSubDomain().GetDimensions(); ++d) {
        stream << " " << cg->centroid[d];
    }

    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
}
PetscErrorCode ablate::boundarySolver::DebugBoundarySolver::ComputeRHSFunction(PetscReal time, Vec locXVec, Vec locFVec) {
    PetscFunctionBeginUser;
    PetscCall(BoundarySolver::ComputeRHSFunction(time, locXVec, locFVec));

    // Get the rank
    PetscMPIInt rank;
    MPI_Comm_rank(GetSubDomain().GetComm(), &rank) >> utilities::PetscUtilities::checkError;

    auto solverDirectory = ablate::environment::RunEnvironment::Get().GetOutputDirectory() / GetSolverId();

    std::ofstream stencilFile;
    stencilFile.open(solverDirectory / ("source_rank_" + std::to_string(rank) + "." + std::to_string(time) + ".txt"));

    // Output the header info
    stencilFile << "p ";
    switch (GetSubDomain().GetDimensions()) {
        case 1:
            stencilFile << " x";
            break;
        case 2:
            stencilFile << " x y";
            break;
        case 3:
            stencilFile << " x y z";
            break;
    }

    // Get the fields
    for (const auto& field : GetSubDomain().GetFields()) {
        // value
        if (field.components.empty()) {
            stencilFile << " " << field.name;
        } else {
            for (const auto& compName : field.components) {
                stencilFile << " " << compName;
            }
        }
        // source
        if (field.components.empty()) {
            stencilFile << " s_" << field.name;
        } else {
            for (const auto& compName : field.components) {
                stencilFile << " s_" << compName;
            }
        }
    }
    stencilFile << std::endl;

    // Get the region to march over
    // Get pointers to sol, aux, and f vectors
    const PetscScalar* locXArray;
    PetscCall(VecGetArrayRead(locXVec, &locXArray));

    const PetscScalar* locFArray;
    PetscCall(VecGetArrayRead(locFVec, &locFArray));

    // March over each cell in this region
    for (const auto& stencilInfo : gradientStencils) {
        // output the cell geom
        OutputStencilCellLocation(stencilFile, stencilInfo.cellId);

        // get each of the fields
        for (const auto& field : GetSubDomain().GetFields()) {
            const PetscScalar* localXValues;
            PetscCall(DMPlexPointLocalFieldRead(GetSubDomain().GetDM(), stencilInfo.cellId, field.id, locXArray, &localXValues));
            for (PetscInt c = 0; c < field.numberComponents; ++c) {
                stencilFile << " " << std::setprecision(16) << localXValues[c];
            }

            const PetscScalar* localFValues;
            PetscCall(DMPlexPointLocalFieldRead(GetSubDomain().GetDM(), stencilInfo.cellId, field.id, locFArray, &localFValues));
            for (PetscInt c = 0; c < field.numberComponents; ++c) {
                stencilFile << " " << std::setprecision(16) << localFValues[c];
            }
        }
        stencilFile << std::endl;

        // Now output each stencil location
        for (PetscInt s = 0; s < stencilInfo.stencilSize; s++) {
            OutputStencilCellLocation(stencilFile, stencilInfo.stencil[s]);

            // get each of the fields
            for (const auto& field : GetSubDomain().GetFields()) {
                const PetscScalar* localXValues;
                PetscCall(DMPlexPointLocalFieldRead(GetSubDomain().GetDM(), stencilInfo.stencil[s], field.id, locXArray, &localXValues));
                for (PetscInt c = 0; c < field.numberComponents; ++c) {
                    stencilFile << " " << std::setprecision(16) << localXValues[c];
                }

                const PetscScalar* localFValues;
                PetscCall(DMPlexPointLocalFieldRead(GetSubDomain().GetDM(), stencilInfo.stencil[s], field.id, locFArray, &localFValues));
                for (PetscInt c = 0; c < field.numberComponents; ++c) {
                    stencilFile << " " << std::setprecision(16) << localFValues[c];
                }
            }
            stencilFile << std::endl;
        }
    }

    // clean up access
    PetscCall(VecRestoreArrayRead(locXVec, &locXArray));
    PetscCall(VecRestoreArrayRead(locFVec, &locFArray));
    stencilFile.close();
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::boundarySolver::DebugBoundarySolver, "A debug version of the solver used to compute boundary values in boundary cells",
         ARG(std::string, "id", "the name of the flow field"), ARG(ablate::domain::Region, "region", "the region to apply this solver."),
         ARG(ablate::domain::Region, "fieldBoundary", "the region describing the faces between the boundary and field"),
         ARG(std::vector<ablate::boundarySolver::BoundaryProcess>, "processes", "a list of boundary processes"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"),
         OPT(bool, "mergeFaces", "determine if multiple faces should be merged for a single cell, default if false"));
