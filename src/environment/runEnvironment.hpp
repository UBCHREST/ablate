#ifndef ABLATELIBRARY_RUNENVIRONMENT_HPP
#define ABLATELIBRARY_RUNENVIRONMENT_HPP
#include <filesystem>
#include <memory>
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

   private:
    inline static std::unique_ptr<RunEnvironment> runEnvironment = std::unique_ptr<RunEnvironment>();
};
}  // namespace ablate::environment

#endif  // ABLATELIBRARY_RUNENVIRONMENT_H
