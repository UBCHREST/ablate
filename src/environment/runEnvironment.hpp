#ifndef ABLATELIBRARY_RUNENVIRONMENT_HPP
#define ABLATELIBRARY_RUNENVIRONMENT_HPP
#include <filesystem>
#include <functional>
#include <memory>
#include <parameters/mapParameters.hpp>
#include <regex>
#include <string>
#include "parameters/parameters.hpp"

namespace ablate::environment {
class RunEnvironment {
   private:
    //! Store default arguments
    inline static int DefaultGlobalArgc = 0;

    //! Store default arguments
    inline static char** DefaultGlobalArgs = nullptr;

    //! Store the global main arg count
    inline static int* GlobalArgc = &DefaultGlobalArgc;

    //! Store the global main args
    inline static char*** GlobalArgs = &DefaultGlobalArgs;

    //! the directory to store output
    std::filesystem::path outputDirectory;

    //! the title of the simulation
    const std::string title;

    // default empty funEnvironment
    explicit RunEnvironment();

    //! store known directory variables
    static const inline std::regex OutputDirectoryVariable = std::regex("\\$OutputDirectory");

    /**
     * Struct to hold the name and function to be called in first in/first out clean up order
     */
    struct FinalizeFunction {
        std::string name;
        std::function<void()> function;
    };

    /**
     * functions to be called externally by the finalize first in/first out function
     */
    inline static std::vector<FinalizeFunction> finalizeFunctions;

   public:
    explicit RunEnvironment(const parameters::Parameters&, const std::filesystem::path& inputPath = {});
    ~RunEnvironment() = default;

    // force RunEnvironment to be a singleton
    RunEnvironment(RunEnvironment& other) = delete;
    void operator=(const RunEnvironment&) = delete;

    // create an empty run env
    static void Setup();

    // static access methods
    static void Setup(const parameters::Parameters&, const std::filesystem::path& inputPath = {});
    inline static const RunEnvironment& Get() {
        if (!runEnvironment) {
            runEnvironment.reset(new RunEnvironment());
        }
        return *runEnvironment;
    }

    /**
     * Return the path to the root of the current output directory
     * @return
     */
    [[nodiscard]] inline const std::filesystem::path& GetOutputDirectory() const { return outputDirectory; }

    /**
     * replaces any known runtime variables with known values
     *  supported values:
     *      - $OutputDirectory is replaced with outputDirectory
     * @param value
     * @return
     */
    void ExpandVariables(std::string& value) const;

    /**
     * initialize ablate
     */
    static void Initialize(int* argc, char*** args);

    /**
     * function to register cleanup function
     */
    static void RegisterCleanUpFunction(const std::string& name, std::function<void()>);

    /**
     * Last thing that any program should do is cleanup
     * @param name
     */
    static void Finalize();

    static inline int* GetArgCount() { return GlobalArgc; }

    static inline char*** GetArgs() { return GlobalArgs; }

    /**
     * Return the current version as a string_view to standardize access to the version
     */
    static std::string_view GetVersion();

    /**
     * Run time parameters is a simple class that helps set RunEnvironmentParameters when not using an input file
     */
    class Parameters : public parameters::MapParameters {
       public:
        /**
         * directly set the output directory for the parameters
         * @return
         */
        inline Parameters& OutputDirectory(const std::filesystem::path& path) {
            Insert("directory", path.string());
            return *this;
        }

        /**
         * turn on directory tagging
         * @return
         */
        inline Parameters& TagDirectory(bool tag) {
            Insert("tagDirectory", tag);
            return *this;
        }
        /**
         * set the name for the simulation
         * @return
         */
        inline Parameters& Title(std::string simulationTitle) {
            Insert("title", simulationTitle);
            return *this;
        }
    };

   private:
    inline static std::unique_ptr<RunEnvironment> runEnvironment = std::unique_ptr<RunEnvironment>();
};
}  // namespace ablate::environment

#endif  // ABLATELIBRARY_RUNENVIRONMENT_HPP
