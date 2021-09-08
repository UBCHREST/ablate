#include "yamlRestartManager.hpp"
#include <utilities/petscError.hpp>
#include "runEnvironment.hpp"
#include "utilities/mpiError.hpp"
#include <memory>
#include <fstream>
#include <yaml-cpp/yaml.h>

void ablate::environment::YamlRestartManager::Register(std::weak_ptr<Restartable> restartable) {
    // store the pointer to the restartable object
    restartables.push_back(restartable);

    // if there is a restore point, look up the data and restore
    if(restoreNode){
        if(auto restartableObject = restartable.lock()) {
            const auto& name = restartableObject->GetName();
            const auto path = restoreDirectory / name;

            // Get the child node
            auto childNode = restoreNode[name];

            // Create the new restore state
            YamlRestoreState restoreState(childNode, path);

            // Restore the state
            restartableObject->Restore(restoreState);
        }
    }

}

PetscErrorCode ablate::environment::YamlRestartManager::YamlRestartManagerSaveStateFunction(TS ts, PetscInt steps, PetscReal time, Vec u, void* mctx) {
    PetscFunctionBeginUser;

    YamlRestartManager* yamlRestartManager = (YamlRestartManager*)mctx;

    try {

        // determine the save directory
        auto saveDirectory = environment::RunEnvironment::Get().GetOutputDirectory()/restartDirectoryName;

        YAML::Emitter out;
        out << YAML::BeginMap;
        out << YAML::Key << "restoreDirectory";
        out << YAML::Value << absolute(saveDirectory);

        for(std::weak_ptr<ablate::environment::Restartable>& restartableWeakPtr : yamlRestartManager->restartables){
            if(auto restartable = restartableWeakPtr.lock()){
                const auto& name = restartable->GetName();
                const auto path = saveDirectory/name;

                // prepare the emitter
                out << YAML::Key << name;
                out << YAML::Value << YAML::BeginMap;

                // Create the new save state
                YamlSaveState saveState(out, path);

                // Save the state
                restartable->Save(saveState);
                out << YAML::EndMap;
            }
        }

        // write to file if we are on the zero rank
        int rank;
        MPI_Comm_rank(PetscObjectComm((PetscObject)ts), &rank) >> checkMpiError;
        if (rank == 0) {
            auto restartFilePath = environment::RunEnvironment::Get().GetOutputDirectory() / "restart.rst";
            std::ofstream restartFile;
            restartFile.open(restartFilePath);
            restartFile << out.c_str();
            restartFile.close();
        }

    } catch (std::exception& exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exception.what());
    }

    PetscFunctionReturn(0);
}

ablate::environment::YamlRestartManager::YamlRestartManager(YAML::Node restoreNodeIn) : restoreDirectory(restoreNodeIn.IsDefined() ? std::filesystem::path(restoreNodeIn["restoreDirectory"].as<std::string>()) : std::filesystem::path()), restoreNode(restoreNodeIn) {}

ablate::environment::YamlRestartManager::YamlRestartManager(std::filesystem::path restoreFile): YamlRestartManager(restoreFile.empty()? YAML::Node(YAML::NodeType::Undefined) : YAML::LoadFile(restoreFile)) {}
std::filesystem::path ablate::environment::YamlRestartManager::GetInputPath() const {
    if(restoreNode){
        return restoreNode["inputPath"].as<std::string>();
    }
    return {};
}

ablate::environment::YamlRestartManager::YamlSaveState::YamlSaveState(YAML::Emitter& yamlEmitter, std::filesystem::path saveDirectory) : yamlEmitter(yamlEmitter), saveDirectory(saveDirectory) {}
void ablate::environment::YamlRestartManager::YamlSaveState::Save(const std::string& name, const std::string& value) {
    yamlEmitter << YAML::Key << name;
    yamlEmitter << YAML::Key << value;
}

void ablate::environment::YamlRestartManager::YamlSaveState::Save(const std::string& name, Vec vec) {
    // Create the output path
    auto outputFilePath = saveDirectory/(name + ".bin");

    // Make sure that the output directory exists
    int rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)vec), &rank) >> checkMpiError;
    if (rank == 0) {
        std::filesystem::create_directories(saveDirectory);
    }

    // write to file if we are on the zero rank
    PetscViewer petscViewer = nullptr;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, outputFilePath.string().c_str(), FILE_MODE_WRITE, &petscViewer) >> checkError;
    VecView(vec, petscViewer) >> checkError;
    PetscViewerDestroy(&petscViewer) >> checkError;
}

ablate::environment::YamlRestartManager::YamlRestoreState::YamlRestoreState(YAML::Node& yamlNode, std::filesystem::path saveDirectory): yamlNode(yamlNode), saveDirectory(saveDirectory) {}
std::optional<std::string> ablate::environment::YamlRestartManager::YamlRestoreState::GetString(std::string paramName) const {
    if(yamlNode.IsMap()){
        auto value = yamlNode[paramName];
        if(value){
            return value.as<std::string>();
        }else{
            return {};
        }
    }else{
        return {};
    }
}
std::unordered_set<std::string> ablate::environment::YamlRestartManager::YamlRestoreState::GetKeys() const {
    std::unordered_set<std::string> keys;

    for (auto childNode : yamlNode) {
        keys.insert(key_to_string(childNode.first));
    }

    return keys;
}

void ablate::environment::YamlRestartManager::YamlRestoreState::Get(const std::string& name, Vec vec) const {
    // Create the output path
    auto outputFilePath = saveDirectory/(name + ".bin");

    // Make sure that the output directory exists
    int rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)vec), &rank) >> checkMpiError;
    if (rank == 0) {
        std::filesystem::create_directories(saveDirectory);
    }

    // Load the saved vector
    PetscViewer petscViewer = nullptr;
    PetscViewerBinaryOpen(PetscObjectComm((PetscObject)vec), outputFilePath.c_str(), FILE_MODE_READ, &petscViewer) >> checkError;
    VecLoad(vec, petscViewer) >> checkError;
    PetscViewerDestroy(&petscViewer) >> checkError;
}

#include "parser/registrar.hpp"
REGISTERDEFAULT(ablate::environment::RestartManager, ablate::environment::YamlRestartManager, "default yaml restart manager", OPT(std::filesystem::path, "restartFile", "Path to an existing restart file.  If not provided the the system does not restart."));
