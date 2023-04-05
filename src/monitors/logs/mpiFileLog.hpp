#ifndef ABLATELIBRARY_MPIFILELOG_HPP
#define ABLATELIBRARY_MPIFILELOG_HPP

#include <petsc.h>
#include <filesystem>
#include "log.hpp"

namespace ablate::monitors::logs {

class MpiFileLog : public Log {
   private:
    std::filesystem::path outputPath;
    FILE *file = nullptr;

   public:
    explicit MpiFileLog(std::string fileName);
    ~MpiFileLog() override;
    // allow access to all print from base
    using Log::Print;
    void Printf(const char *, ...) final;
    void Initialize(MPI_Comm comm) final;
};
}  // namespace ablate::monitors::logs

#endif  // ABLATELIBRARY_MPIFILELOG_HPP
