#ifndef ABLATELIBRARY_YAMLRESTARTMANAGER_HPP
#define ABLATELIBRARY_YAMLRESTARTMANAGER_HPP

#include <filesystem>
#include <memory>
#include <vector>
#include "restartManager.hpp"
#include "restartable.hpp"
#include "yaml-cpp/emitter.h"
#include "yaml-cpp/node/node.h"

namespace ablate::environment {

class YamlRestartManager : public RestartManager {
   private:
    std::vector<std::weak_ptr<ablate::environment::Restartable>> restartables;

    // Petsc function used to save the system state
    static PetscErrorCode YamlRestartManagerSaveStateFunction(TS ts, PetscInt steps, PetscReal time, Vec u, void* mctx);

    // store references to the restore directory
    inline const static std::string restartDirectoryName = "restart";
    const std::filesystem::path restoreDirectory;

    // Store a YAML::NODE is we are in restore state
    YAML::Node restoreNode;

    // Create a private class to save to the yaml file
    class YamlSaveState : public ablate::environment::SaveState{
       private:
        YAML::Emitter& yamlEmitter;
        const std::filesystem::path saveDirectory;
       public:
        YamlSaveState(YAML::Emitter& yamlEmitter, std::filesystem::path saveDirectory);
        void Save(const std::string&, const std::string&) final;
        void Save(const std::string&, Vec) final;
    };

    // Create a private class to save to read from the yaml node
    class YamlRestoreState : public ablate::environment::RestoreState{
       private:
        YAML::Node& yamlNode;
        const std::filesystem::path saveDirectory;
       public:
        YamlRestoreState(YAML::Node& yamlNode, std::filesystem::path saveDirectory);
        std::optional<std::string> GetString(std::string paramName) const final;
        std::unordered_set<std::string> GetKeys() const final;
        void Get(const std::string&, Vec) const final;
    };


   public:
    explicit YamlRestartManager(YAML::Node restoreNode);

    explicit YamlRestartManager(std::filesystem::path restoreFile = {});

    /**
     * The Register method registers the class to be saved and restores any values if present.
     */
    void Register(std::weak_ptr<Restartable>) override;

    /**
     * Return the function that is called to save state
     * @return
     */
    PetscSaveStateFunction GetTSFunction() override {return YamlRestartManagerSaveStateFunction;}

    /**
     * Return the input path if we are in restore state
     * @return
     */
    std::filesystem::path GetInputPath() const override;


};

}  // namespace ablate::environment
#endif  // ABLATELIBRARY_YAMLRESTARTMANAGER_HPP
