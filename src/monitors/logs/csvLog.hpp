#ifndef ABLATELIBRARY_CSVLOG_HPP
#define ABLATELIBRARY_CSVLOG_HPP

#include <petsc.h>
#include <filesystem>
#include "log.hpp"

namespace ablate::monitors::logs {

class CsvLog : public Log {
   private:
    std::filesystem::path outputPath;
    FILE* file = nullptr;
    MPI_Comm comm = MPI_COMM_SELF;
    const char* separator = ",";

   public:
    explicit CsvLog(std::string fileName);
    ~CsvLog() override;

    void Initialize(MPI_Comm comm) final;

    // allow access to all print from base
    using Log::Print;

    /**
     * print all arguments as a new line in a csv
     * @param ...
     */
    void Printf(const char*, ...) final;

    /**
     * print statements are not printed to the csv
     * @param value
     */
    void Print(const char* value) final{};

    /**
     * print vectors are appended to the current line
     * @param name
     * @param num
     * @param format
     */
    void Print(const char* name, std::size_t num, const double*, const char* format = nullptr) final;
};
}  // namespace ablate::monitors::logs

#endif  // ABLATELIBRARY_CSVLOG_HPP
