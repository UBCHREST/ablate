#include "hdf5MultiFileSerializer.hpp"
#include <petscviewerhdf5.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>
#include <utility>
#include "environment/runEnvironment.hpp"
#include "generators.hpp"

ablate::io::Hdf5MultiFileSerializer::Hdf5MultiFileSerializer(std::shared_ptr<ablate::io::interval::Interval> interval, const std::shared_ptr<parameters::Parameters>& options)
    : interval(std::move(interval)), rootOutputDirectory(environment::RunEnvironment::Get().GetOutputDirectory()) {
    // Load the metadata from the file is available, otherwise set to 0
    auto restartFilePath = rootOutputDirectory / "restart.rst";

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

    // setup petsc options if provided
    if (options) {
        PetscOptionsCreate(&petscOptions) >> utilities::PetscUtilities::checkError;
        options->Fill(petscOptions);
    }
}

ablate::io::Hdf5MultiFileSerializer::~Hdf5MultiFileSerializer() {
    // save each serializer
    for (const std::string& id : postProcessesIds) {
        std::vector<std::filesystem::path> inputFilePaths;

        auto directoryPath = GetOutputDirectoryPath(id);
        for (const auto& file : std::filesystem::directory_iterator(directoryPath)) {
            if (file.path().extension() == ".hdf5") {
                inputFilePaths.push_back(file.path());
            }
        }

        // sort the paths
        std::sort(inputFilePaths.begin(), inputFilePaths.end());

        // run the convert function
        std::filesystem::path outputFile = directoryPath / (id + ".xmf");
        xdmfGenerator::Generate(inputFilePaths, outputFile);
    }

    if (petscOptions) {
        ablate::utilities::PetscUtilities::PetscOptionsDestroyAndCheck("ablate::io::Hdf5MultiFileSerializer::Hdf5MultiFileSerializer", &petscOptions);
    }
}

void ablate::io::Hdf5MultiFileSerializer::Register(std::weak_ptr<Serializable> serializable) {
    serializables.push_back(serializable);

    // Mark this to clean up
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank) >> utilities::PetscUtilities::checkError;

    if (auto serializableObject = serializable.lock()) {
        // resume if needed
        if (resumed) {
            PetscViewer petscViewer = nullptr;
            StartEvent("PetscViewerHDF5Open");
            switch (serializableObject->Serialize()) {
                case Serializable::SerializerType::collective: {
                    auto filePath = GetOutputFilePath(serializableObject->GetId());
                    PetscViewerHDF5Open(PETSC_COMM_WORLD, filePath.string().c_str(), FILE_MODE_UPDATE, &petscViewer) >> utilities::PetscUtilities::checkError;
                } break;
                case Serializable::SerializerType::serial: {
                    auto filePath = GetOutputFilePath(serializableObject->GetId(), rank);
                    PetscViewerHDF5Open(PETSC_COMM_SELF, filePath.string().c_str(), FILE_MODE_UPDATE, &petscViewer) >> utilities::PetscUtilities::checkError;
                } break;
                default:
                    throw std::invalid_argument("Unable to determine Serializer Type");
            }
            EndEvent();

            // set the petsc options if provided
            PetscObjectSetOptions((PetscObject)petscViewer, petscOptions) >> utilities::PetscUtilities::checkError;
            PetscViewerSetFromOptions(petscViewer) >> utilities::PetscUtilities::checkError;
            PetscViewerViewFromOptions(petscViewer, nullptr, "-hdf5ViewerView") >> utilities::PetscUtilities::checkError;

            // Restore the simulation
            StartEvent("Restore");
            // NOTE: as far as the output file the sequence number is always zero because it is a new file
            serializableObject->Restore(petscViewer, 0, time) >> utilities::PetscUtilities::checkError;
            EndEvent();

            StartEvent("PetscViewerHDF5Destroy");
            PetscViewerDestroy(&petscViewer) >> utilities::PetscUtilities::checkError;
            EndEvent();
        } else {
            // Create an output directory
            auto outputDirectory = GetOutputDirectoryPath(serializableObject->GetId());
            if (rank == 0) {
                std::filesystem::create_directory(outputDirectory);
            }
            MPI_Barrier(PETSC_COMM_WORLD);
        }

        if (rank == 0) {
            postProcessesIds.push_back(serializableObject->GetId());
        }
    }
}

PetscErrorCode ablate::io::Hdf5MultiFileSerializer::Hdf5MultiFileSerializerSaveStateFunction(TS ts, PetscInt steps, PetscReal time, Vec, void* ctx) {
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
        TSGetTimeStep(ts, &(hdf5Serializer->dt)) >> utilities::PetscUtilities::checkError;

        // Save this to a file
        hdf5Serializer->SaveMetadata(ts);

        // save each serializer
        for (auto& serializablePtr : hdf5Serializer->serializables) {
            if (auto serializableObject = serializablePtr.lock()) {
                // Create an output path

                PetscViewer petscViewer = nullptr;
                hdf5Serializer->StartEvent("PetscViewerHDF5Open");
                switch (serializableObject->Serialize()) {
                    case Serializable::SerializerType::collective: {
                        auto filePath = hdf5Serializer->GetOutputFilePath(serializableObject->GetId());
                        PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, filePath.string().c_str(), FILE_MODE_WRITE, &petscViewer));
                    } break;
                    case Serializable::SerializerType::serial: {
                        PetscMPIInt rank;
                        MPI_Comm_rank(PetscObjectComm((PetscObject)ts), &rank);
                        auto filePath = hdf5Serializer->GetOutputFilePath(serializableObject->GetId(), rank);
                        PetscCall(PetscViewerHDF5Open(PETSC_COMM_SELF, filePath.string().c_str(), FILE_MODE_WRITE, &petscViewer));
                    } break;
                    default:
                        throw std::invalid_argument("Unable to determine Serializer Type");
                }
                hdf5Serializer->EndEvent();

                // set the petsc options if provided
                PetscCall(PetscObjectSetOptions((PetscObject)petscViewer, hdf5Serializer->petscOptions));
                PetscCall(PetscViewerSetFromOptions(petscViewer));
                PetscCall(PetscViewerViewFromOptions(petscViewer, nullptr, "-hdf5ViewerView"));

                hdf5Serializer->StartEvent("Save");
                // NOTE: as far as the output file the sequence number is always zero because it is a new file
                PetscCall(serializableObject->Save(petscViewer, 0, time));
                hdf5Serializer->EndEvent();

                hdf5Serializer->StartEvent("PetscViewerHDF5Destroy");
                PetscCall(PetscViewerDestroy(&petscViewer));
                hdf5Serializer->EndEvent();
            }
        }
    }
    PetscFunctionReturn(0);
}

void ablate::io::Hdf5MultiFileSerializer::SaveMetadata(TS ts) const {
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
    MPI_Comm_rank(PetscObjectComm((PetscObject)ts), &rank) >> utilities::PetscUtilities::checkError;
    if (rank == 0) {
        auto restartFilePath = rootOutputDirectory / "restart.rst";
        // keep a back of the restart file incase writing fails
        if (exists(restartFilePath)) {
            auto backupRestartFilePath = rootOutputDirectory / "restart.bak";
            try {
                std::filesystem::copy(restartFilePath, backupRestartFilePath, std::filesystem::copy_options::overwrite_existing);
            } catch (const std::filesystem::filesystem_error& error) {
                std::cout << "Cannot create restart.bak file: " << error.what() << " Continuing without backup file." << std::endl;
            }
        }
        std::ofstream restartFile;
        restartFile.open(restartFilePath);
        restartFile << out.c_str();
        restartFile.close();
    }
    PetscFunctionReturnVoid();
}

void ablate::io::Hdf5MultiFileSerializer::RestoreTS(TS ts) {
    if (resumed) {
        TSSetStepNumber(ts, timeStep);
        TSSetTime(ts, time);
        TSSetTimeStep(ts, dt);
    }
}

std::filesystem::path ablate::io::Hdf5MultiFileSerializer::GetOutputFilePath(const std::string& objectId) const {
    std::stringstream sequenceNumberOutputStream;
    sequenceNumberOutputStream << std::setw(5) << std::setfill('0') << sequenceNumber;
    auto sequenceNumberOutputString = "." + sequenceNumberOutputStream.str();
    return GetOutputDirectoryPath(objectId) / (objectId + sequenceNumberOutputString + extension);
}

std::filesystem::path ablate::io::Hdf5MultiFileSerializer::GetOutputDirectoryPath(const std::string& objectId) const { return rootOutputDirectory / objectId; }
std::filesystem::path ablate::io::Hdf5MultiFileSerializer::GetOutputFilePath(const std::string& objectId, int rank) const {
    std::stringstream sequenceNumberOutputStream;
    sequenceNumberOutputStream << "." << std::setw(5) << std::setfill('0') << rank;
    sequenceNumberOutputStream << "." << std::setw(5) << std::setfill('0') << sequenceNumber;
    return GetOutputDirectoryPath(objectId) / (objectId + sequenceNumberOutputStream.str() + extension);
}
#include "registrar.hpp"
REGISTER(ablate::io::Serializer, ablate::io::Hdf5MultiFileSerializer, "serializer for IO that writes each time to a separate hdf5 file",
         ARG(ablate::io::interval::Interval, "interval", "The interval object used to determine write interval."),
         OPT(ablate::parameters::Parameters, "options", "options for the viewer passed directly to PETSc including (hdf5ViewerView, viewer_hdf5_collective, viewer_hdf5_sp_output"));
