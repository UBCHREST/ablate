#ifndef ABLATELIBRARY_RUNENVIRONMENT_HPP
#define ABLATELIBRARY_RUNENVIRONMENT_HPP
#include <filesystem>
#include <memory>
#include <string>
#include "parameters/parameters.hpp"

namespace ablate::environment {
class RunEnvironment {
   private:
    std::filesystem::path outputDirectory;
    const std::string title;
    std::filesystem::path inputPath;

    // default empty funEnvironment
    explicit RunEnvironment();

   public:
    explicit RunEnvironment(const parameters::Parameters&, std::filesystem::path inputPath = {});
    ~RunEnvironment() = default;

    // force RunEnvironment to be a singleton
    RunEnvironment(RunEnvironment& other) = delete;
    void operator=(const RunEnvironment&) = delete;

    // static access methods
    static void Setup(const parameters::Parameters&, std::filesystem::path inputPath = {});
    inline static const RunEnvironment& Get() {
        if (!runEnvironment) {
            runEnvironment.reset(new RunEnvironment());
        }
        return *runEnvironment;
    }

    inline const std::filesystem::path& GetOutputDirectory() const { return outputDirectory; }
    inline const std::filesystem::path& GetInputPath() const { return inputPath; }

   private:
    inline static std::unique_ptr<RunEnvironment> runEnvironment = std::unique_ptr<RunEnvironment>();
};
}  // namespace ablate::environment

#endif  // ABLATELIBRARY_RUNENVIRONMENT_H
