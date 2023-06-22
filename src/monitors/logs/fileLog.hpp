#ifndef ABLATELIBRARY_FILELOG_HPP
#define ABLATELIBRARY_FILELOG_HPP

#include <petsc.h>
#include <filesystem>
#include "log.hpp"

namespace ablate::monitors::logs {

class FileLog : public Log {
   private:
    std::filesystem::path outputPath;
    FILE *file = nullptr;
    MPI_Comm comm = MPI_COMM_SELF;

   public:
    explicit FileLog(std::string fileName);
    ~FileLog() override;
    // allow access to all print from base
    using Log::Print;
    void Printf(const char *, ...) final;
    void Initialize(MPI_Comm comm) final;
};
}  // namespace ablate::monitors::logs

#endif  // ABLATELIBRARY_FILELOG_HPP
