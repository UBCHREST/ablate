#include "restart.hpp"
#include <petscviewerhdf5.h>
#include <yaml-cpp/emitter.h>
#include <environment/runEnvironment.hpp>
#include <fstream>

PetscErrorCode ablate::monitors::Restart::OutputRestart(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx) {
    PetscFunctionBeginUser;

    auto monitor = (ablate::monitors::Restart *)mctx;
    if (monitor->interval == 0 || (steps % monitor->interval == 0)) {
        // By default save the ts restart parameters.  This code corresponds to the read in the time stepper
        PetscErrorCode ierr;
        // create the output file path
        auto outputFilePath = environment::RunEnvironment::Get().GetOutputDirectory() / "restart.solution.bin";

        // setup the petsc viewer
        PetscViewer petscViewer = nullptr;
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, outputFilePath.string().c_str(), FILE_MODE_WRITE, &petscViewer);
        CHKERRQ(ierr);
        ierr = VecView(u, petscViewer);
        CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&petscViewer);
        CHKERRQ(ierr);

        // get some output information
        PetscReal dt;
        ierr = TSGetTimeStep(ts, &dt);
        CHKERRQ(ierr);

        // output the restart information
        YAML::Emitter out;
        out << YAML::BeginMap;
        out << YAML::Key << "inputPath";
        out << YAML::Value << environment::RunEnvironment::Get().GetInputPath();

        // Output the ts specific restart information
        out << YAML::Key << "ts";
        out << YAML::BeginMap;
        out << YAML::Key << "steps";
        out << YAML::Value << steps;
        out << YAML::Key << "time";
        out << YAML::Value << time;
        out << YAML::Key << "dt";
        out << YAML::Value << dt;
        out << YAML::Key << "solutionVec";
        out << YAML::Value << outputFilePath;
        out << YAML::EndMap;
        out << YAML::EndMap;

        // write to file if we are on the zero rank
        int rank;
        int mpiErr = MPI_Comm_rank(PetscObjectComm((PetscObject)ts), &rank);
        CHKERRMPI(mpiErr);
        if (rank == 0) {
            auto restartFilePath = environment::RunEnvironment::Get().GetOutputDirectory() / "restart.rst";
            std::ofstream restartFile;
            restartFile.open(restartFilePath);
            restartFile << out.c_str();
            restartFile.close();
        }
    }

    PetscFunctionReturn(0);
}

ablate::monitors::Restart::Restart(int interval) : interval(interval) {}

#include "parser/registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::Restart, "outputs the required files needed to restart the simulation", ARG(int, "interval", "how often to write the restart files"));
