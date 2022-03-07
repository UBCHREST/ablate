#include "hdf5MultiFileSerializer.hpp"
#include <petscviewerhdf5.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include "environment/runEnvironment.hpp"

ablate::io::Hdf5MultiFileSerializer::Hdf5MultiFileSerializer(std::shared_ptr<ablate::io::interval::Interval> interval) : interval(interval) {
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
void ablate::io::Hdf5MultiFileSerializer::Register(std::weak_ptr<Serializable> serializable) { serializables.push_back(serializable); }

PetscErrorCode ablate::io::Hdf5MultiFileSerializer::Hdf5MultiFileSerializerSaveStateFunction(TS ts, PetscInt steps, PetscReal time, Vec u, void* ctx) {
    PetscFunctionBeginUser;
    auto hdf5Serializer = (Hdf5MultiFileSerializer*)ctx;

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

        std::stringstream sequenceNumberOutputStream;
        sequenceNumberOutputStream << std::setw(5) << std::setfill('0') << hdf5Serializer->sequenceNumber;
        auto sequenceNumberOutputString = "." + sequenceNumberOutputStream.str();

        try {
            // save each serializer
            for (auto& serializablePtr : hdf5Serializer->serializables) {
                if (auto serializableObject = serializablePtr.lock()) {
                    // Create an output path
                    auto filePath = environment::RunEnvironment::Get().GetOutputDirectory() / (serializableObject->GetId() + sequenceNumberOutputString + extension);

                    PetscViewer petscViewer = nullptr;

                    hdf5Serializer->StartEvent("PetscViewerHDF5Open");
                    PetscViewerHDF5Open(PETSC_COMM_WORLD, filePath.string().c_str(), FILE_MODE_WRITE, &petscViewer) >> checkError;
                    hdf5Serializer->EndEvent();

                    hdf5Serializer->StartEvent("Save");
                    serializableObject->Save(petscViewer, 0, time);
                    hdf5Serializer->EndEvent();

                    hdf5Serializer->StartEvent("PetscViewerHDF5Destroy");
                    PetscViewerDestroy(&petscViewer) >> checkError;
                    hdf5Serializer->EndEvent();
                }
            }
        } catch (std::exception& exception) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exception.what());
        }
    }
    PetscFunctionReturn(0);
}

void ablate::io::Hdf5MultiFileSerializer::SaveMetadata(TS ts) {
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
    MPI_Comm_rank(PetscObjectComm((PetscObject)ts), &rank) >> checkError;
    if (rank == 0) {
        auto restartFilePath = environment::RunEnvironment::Get().GetOutputDirectory() / "restart.rst";
        std::ofstream restartFile;
        restartFile.open(restartFilePath);
        restartFile << out.c_str();
        restartFile.close();
    }
}

void ablate::io::Hdf5MultiFileSerializer::RestoreTS(TS ts) {
    if (resumed) {
        TSSetStepNumber(ts, timeStep);
        TSSetTime(ts, time);
        TSSetTimeStep(ts, dt);
    }
}

#include "registrar.hpp"
REGISTER(ablate::io::Serializer, ablate::io::Hdf5MultiFileSerializer, "serializer for IO that writes each time to a separate hdf5 file",
         ARG(ablate::io::interval::Interval, "interval", "The interval object used to determine write interval."));
