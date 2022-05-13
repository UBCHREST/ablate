#ifndef ABLATELIBRARY_RUNENVIRONMENT_HPP
#define ABLATELIBRARY_RUNENVIRONMENT_HPP
#include <filesystem>
#include <memory>
#include <regex>
#include <string>
#include "parameters/parameters.hpp"

namespace ablate::environment {
class RunEnvironment {
   private:
    std::filesystem::path outputDirectory;
    const std::string title;

    // default empty funEnvironment
    explicit RunEnvironment();

    //! store known directory variables
    static const inline std::regex OutputDirectoryVariable = std::regex("\\$OutputDirectory");

   public:
    explicit RunEnvironment(const parameters::Parameters&, std::filesystem::path inputPath = {});
    ~RunEnvironment() = default;

    // force RunEnvironment to be a singleton
    RunEnvironment(RunEnvironment& other) = delete;
    void operator=(const RunEnvironment&) = delete;

    // create an empty run env
    static void Setup();

    // static access methods
    static void Setup(const parameters::Parameters&, std::filesystem::path inputPath = {});
    inline static const RunEnvironment& Get() {
        if (!runEnvironment) {
            runEnvironment.reset(new RunEnvironment());
        }
        return *runEnvironment;
    }

    inline const std::filesystem::path& GetOutputDirectory() const { return outputDirectory; }

    /**
     * replaces any known runtime variables with known values
     *  supported values:
     *      - $OutputDirectory is replaced with outputDirectory
     * @param value
     * @return
     */
    void ExpandVariables(std::string& value) const;

   private:
    inline static std::unique_ptr<RunEnvironment> runEnvironment = std::unique_ptr<RunEnvironment>();
};
}  // namespace ablate::environment

#endif  // ABLATELIBRARY_RUNENVIRONMENT_H
