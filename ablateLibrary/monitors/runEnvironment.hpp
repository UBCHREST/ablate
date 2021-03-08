#ifndef ABLATELIBRARY_RUNENVIRONMENT_HPP
#define ABLATELIBRARY_RUNENVIRONMENT_HPP
#include "parameters/parameters.hpp"
#include <filesystem>

namespace ablate::monitors {
class RunEnvironment {
   private:
    std::filesystem::path outputDirectory;
    const std::string title;

   public:
    explicit RunEnvironment(std::filesystem::path inputPath,  const parameters::Parameters&);

    // force RunEnvironment to be a singleton
    RunEnvironment(RunEnvironment &other) = delete;
    void operator=(const RunEnvironment &) = delete;

    // static access methods
    static void Setup(std::filesystem::path inputPath,  const parameters::Parameters&);
    inline static const RunEnvironment& Get(){
        return *runEnvironment;
    }

    inline const std::filesystem::path& GetOutputDirectory() const{
        return outputDirectory;
    }

   private:
    inline static std::unique_ptr<RunEnvironment> runEnvironment = nullptr;
};
}

#endif  // ABLATELIBRARY_RUNENVIRONMENT_H
