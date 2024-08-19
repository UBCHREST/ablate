#include "hdf5Serializer.hpp"
#include <petscviewerhdf5.h>
#include <yaml-cpp/yaml.h>
#include <environment/runEnvironment.hpp>
#include <fstream>
#include <io/interval/interval.hpp>
#include <iostream>
#include <utility>
#include "generators.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscUtilities.hpp"

ablate::io::Hdf5Serializer::Hdf5Serializer(std::shared_ptr<ablate::io::interval::Interval> interval) : interval(std::move(interval)) {
    // Load the metadata from the file is available, otherwise set to 0
    auto restartFilePath = environment::RunEnvironment::Get().GetOutputDirectory() / "restart.rst";

    if (std::filesystem::exists(restartFilePath)) {
        resumed = true;
        auto yaml = YAML::LoadFile(restartFilePath);
        time = yaml["time"].as<PetscReal>();
        dt = yaml["dt"].as<PetscReal>();
        timeStep = yaml["timeStep"].as<PetscInt>();
        sequenceNumber = yaml["sequenceNumber"].as<PetscInt>();

        // check for restart info warning
        auto version = yaml["version"];
        if (version.IsDefined()) {
            if (version.as<std::string>() != environment::RunEnvironment::GetVersion()) {
                std::cout << "Warning: Restarting simulation using a different version of ABLATE " << version.as<std::string>() << " vs. " << environment::RunEnvironment::GetVersion();
            }
        }
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
    auto* hdf5Serializer = (Hdf5Serializer*)ctx;

    // Make sure that the same timeStep is not output more than once (this can be from a restart)
    if (steps <= hdf5Serializer->timeStep) {
        PetscFunctionReturn(0);
    }

    if (hdf5Serializer->interval->Check(PetscObjectComm((PetscObject)ts), steps, time)) {
        // Update all metadata
        hdf5Serializer->time = time;
        hdf5Serializer->timeStep = steps;
        hdf5Serializer->sequenceNumber++;
        TSGetTimeStep(ts, &(hdf5Serializer->dt)) >> utilities::PetscUtilities::checkError;

        // Save this to a file
        hdf5Serializer->SaveMetadata(ts);

        // save each serializer
        for (auto& serializer : hdf5Serializer->serializers) {
            PetscCall(serializer->Save(hdf5Serializer->sequenceNumber, time));
        }
    }
    PetscFunctionReturn(0);
}

void ablate::io::Hdf5Serializer::SaveMetadata(TS ts) const {
    PetscFunctionBeginUser;
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
    out << YAML::Key << "version";
    out << YAML::Value << std::string(environment::RunEnvironment::GetVersion());
    out << YAML::EndMap;

    int rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)ts), &rank) >> utilities::MpiUtilities::checkError;
    if (rank == 0) {
        auto restartFilePath = environment::RunEnvironment::Get().GetOutputDirectory() / "restart.rst";
        std::ofstream restartFile;
        restartFile.open(restartFilePath);
        restartFile << out.c_str();
        restartFile.close();
    }
    PetscFunctionReturnVoid();
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
    : serializable(std::move(serializableIn)) {
    if (auto serializableObject = serializable.lock()) {
        MPI_Comm activeComm;
        switch (serializableObject->Serialize()) {
            case Serializable::SerializerType::collective: {
                filePath = environment::RunEnvironment::Get().GetOutputDirectory() / (serializableObject->GetId() + extension);
                activeComm = PETSC_COMM_WORLD;
            } break;
            case Serializable::SerializerType::serial: {
                PetscMPIInt rank;
                MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
                filePath = environment::RunEnvironment::Get().GetOutputDirectory() / (serializableObject->GetId() + "." + std::to_string(rank) + extension);
                activeComm = PETSC_COMM_SELF;
            } break;
            default:
                throw std::invalid_argument("Unable to determine Serializer Type");
        }

        // Check to see if the viewer file exists
        if (resume) {
            if (std::filesystem::exists(filePath)) {
                StartEvent("PetscViewerHDF5Open");
                PetscViewerHDF5Open(activeComm, filePath.string().c_str(), FILE_MODE_UPDATE, &petscViewer) >> utilities::PetscUtilities::checkError;
                EndEvent();

                // Restore the simulation
                StartEvent("Restore");
                serializableObject->Restore(petscViewer, sequenceNumber, time) >> utilities::PetscUtilities::checkError;
                EndEvent();
            } else {
                throw std::runtime_error("Cannot resume simulation.  Unable to locate file: " + filePath.string());
            }
        } else {
            PetscViewerHDF5Open(activeComm, filePath.string().c_str(), FILE_MODE_WRITE, &petscViewer) >> utilities::PetscUtilities::checkError;
        }
    }
}

ablate::io::Hdf5Serializer::Hdf5ObjectSerializer::~Hdf5ObjectSerializer() {
    if (petscViewer) {
        // If this is the root process generate the xdmf file
        PetscMPIInt rank;
        MPI_Comm_rank(PetscObjectComm(PetscObject(petscViewer)), &rank);
        PetscViewerDestroy(&petscViewer) >> utilities::PetscUtilities::checkError;

        if (rank == 0 && !filePath.empty() && std::filesystem::exists(filePath)) {
            xdmfGenerator::Generate(filePath);
        }
    }
}

PetscErrorCode ablate::io::Hdf5Serializer::Hdf5ObjectSerializer::Save(PetscInt sn, PetscReal t) {
    PetscFunctionBeginUser;
    if (auto serializableObject = serializable.lock()) {
        PetscCall(serializableObject->Save(petscViewer, sn, t));
    }
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER_DEFAULT(ablate::io::Serializer, ablate::io::Hdf5Serializer, "default serializer for IO",
                 ARG(ablate::io::interval::Interval, "interval", "The interval object used to determine write interval."));
