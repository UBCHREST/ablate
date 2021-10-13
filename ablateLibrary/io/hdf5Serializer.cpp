#include "hdf5Serializer.hpp"
#include <petscviewerhdf5.h>
#include <yaml-cpp/yaml.h>
#include <environment/runEnvironment.hpp>
#include <fstream>
#include <io/interval/interval.hpp>
#include <utilities/mpiError.hpp>
#include "generators.hpp"
#include "utilities/loggable.hpp"
#include "utilities/petscError.hpp"

ablate::io::Hdf5Serializer::Hdf5Serializer(std::shared_ptr<ablate::io::interval::Interval> interval) : interval(interval) {
    // Load the metadata from the file is available, otherwise set to 0
    auto restartFilePath = environment::RunEnvironment::Get().GetOutputDirectory() / "restart.rst";

    if (std::filesystem::exists(restartFilePath)) {
        resumed = true;
        auto yaml = YAML::LoadFile(restartFilePath);
        time = yaml["time"].as<PetscReal>();
        dt = yaml["dt"].as<PetscReal>();
        timeStep = yaml["timeStep"].as<PetscInt>();
        sequenceNumber = yaml["sequenceNumber"].as<PetscInt>();
    } else {
        resumed = false;
        time = NAN;
        dt = NAN;
        timeStep = -1;
        sequenceNumber = -1;
    }
}

void ablate::io::Hdf5Serializer::Register(std::weak_ptr<Serializable> serializable) {
    // for each serializable object create a Hdf5ObjectSerializer
    serializers.push_back(std::make_unique<Hdf5ObjectSerializer>(serializable, sequenceNumber, time, resumed));
}

PetscErrorCode ablate::io::Hdf5Serializer::Hdf5SerializerSaveStateFunction(TS ts, PetscInt steps, PetscReal time, Vec u, void* ctx) {
    PetscFunctionBeginUser;
    Hdf5Serializer* hdf5Serializer = (Hdf5Serializer*)ctx;

    // Make sure that the same timeStep is not output more than once (this can be from a restart)
    if (steps <= hdf5Serializer->timeStep) {
        PetscFunctionReturn(0);
    }

    if (hdf5Serializer->interval->Check(PetscObjectComm((PetscObject)ts), steps, time)) {
        // Update all metadata
        hdf5Serializer->time = time;
        hdf5Serializer->timeStep = steps;
        hdf5Serializer->sequenceNumber++;
        TSGetTimeStep(ts, &(hdf5Serializer->dt)) >> checkError;

        // Save this to a file
        hdf5Serializer->SaveMetadata(ts);

        try {
            // save each serializer
            for (auto& serializer : hdf5Serializer->serializers) {
                serializer->Save(hdf5Serializer->sequenceNumber, time);
            }
        } catch (std::exception& exception) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exception.what());
        }
    }
    PetscFunctionReturn(0);
}

void ablate::io::Hdf5Serializer::SaveMetadata(TS ts) {
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "time";
    out << YAML::Value << time;
    out << YAML::Key << "dt";
    out << YAML::Value << dt;
    out << YAML::Key << "timeStep";
    out << YAML::Value << timeStep;
    out << YAML::Key << "sequenceNumber";
    out << YAML::Value << sequenceNumber;
    out << YAML::EndMap;

    int rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)ts), &rank) >> checkMpiError;
    if (rank == 0) {
        auto restartFilePath = environment::RunEnvironment::Get().GetOutputDirectory() / "restart.rst";
        std::ofstream restartFile;
        restartFile.open(restartFilePath);
        restartFile << out.c_str();
        restartFile.close();
    }
}

void ablate::io::Hdf5Serializer::RestoreTS(TS ts) {
    if (resumed) {
        TSSetStepNumber(ts, timeStep);
        TSSetTime(ts, time);
        TSSetTimeStep(ts, dt);
    }
}

////////////// Hdf5ObjectSerializer Implementation //////////////
ablate::io::Hdf5Serializer::Hdf5ObjectSerializer::Hdf5ObjectSerializer(std::weak_ptr<Serializable> serializableIn, PetscInt sequenceNumber, PetscReal time, bool resume)
    : serializable(serializableIn) {
    if (auto serializableObject = serializable.lock()) {
        filePath = environment::RunEnvironment::Get().GetOutputDirectory() / (serializableObject->GetId() + extension);

        // Check to see if the viewer file exists
        if (resume) {
            if (std::filesystem::exists(filePath)) {
                StartEvent("PetscViewerHDF5Open");
                PetscViewerHDF5Open(PETSC_COMM_WORLD, filePath.string().c_str(), FILE_MODE_UPDATE, &petscViewer) >> checkError;
                EndEvent();

                // Restore the simulation
                StartEvent("Restore");
                serializableObject->Restore(petscViewer, sequenceNumber, time);
                EndEvent();
            } else {
                throw std::runtime_error("Cannot resume simulation.  Unable to locate file: " + filePath.string());
            }
        } else {
            PetscViewerHDF5Open(PETSC_COMM_WORLD, filePath.string().c_str(), FILE_MODE_WRITE, &petscViewer) >> checkError;
        }
    }
}

ablate::io::Hdf5Serializer::Hdf5ObjectSerializer::~Hdf5ObjectSerializer() {
    if (petscViewer) {
        // If this is the root process generate the xdmf file
        PetscMPIInt rank;
        MPI_Comm_rank(PetscObjectComm(PetscObject(petscViewer)), &rank);
        if (rank == 0 && !filePath.empty() && std::filesystem::exists(filePath)) {
            petscXdmfGenerator::Generate(filePath);
        }

        PetscViewerDestroy(&petscViewer) >> checkError;
    }
}

void ablate::io::Hdf5Serializer::Hdf5ObjectSerializer::Save(PetscInt sn, PetscReal t) {
    if (auto serializableObject = serializable.lock()) {
        serializableObject->Save(petscViewer, sn, t);
    }
}

#include "parser/registrar.hpp"
REGISTERDEFAULT(ablate::io::Serializer, ablate::io::Hdf5Serializer, "default serializer for IO",
                ARG(ablate::io::interval::Interval, "interval", "The interval object used to determine write interval."));
