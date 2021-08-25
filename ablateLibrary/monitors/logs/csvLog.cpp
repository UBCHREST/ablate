#include "csvLog.hpp"
#include <environment/runEnvironment.hpp>
#include <regex>
#include <utilities/petscError.hpp>

static std::regex printfRegex("%[+-]*[0-9]*([.][0-9]+)?[cdefiosuxgCDEFIOUSUXG]");

ablate::monitors::logs::CsvLog::CsvLog(std::string fileName)
    : outputPath(std::filesystem::path(fileName).is_absolute() ? std::filesystem::path(fileName) : ablate::environment::RunEnvironment::Get().GetOutputDirectory() / fileName), file(nullptr) {}

ablate::monitors::logs::CsvLog::~CsvLog() {
    if (file) {
        fclose(file);
    }
}
void ablate::monitors::logs::CsvLog::Initialize(MPI_Comm commIn) {
    Log::Initialize(commIn);
    comm = commIn;
    PetscFOpen(comm, outputPath.c_str(), "w", &file) >> checkError;
}
void ablate::monitors::logs::CsvLog::Printf(const char *format, ...) {
    if (file) {
        // build a new csf format string based upon the formats passed in
        std::string oldFormat(format);
        std::string formatString = "\n";

        std::sregex_iterator begin(oldFormat.begin(), oldFormat.end(), printfRegex);
        std::sregex_iterator end;
        for (std::sregex_iterator i = begin; i != end; ++i) {
            std::smatch match = *i;
            formatString += match.str() + separator;
        }

        va_list args;
        va_start(args, format);
        PetscVFPrintf(file, formatString.c_str(), args) >> checkError;
        va_end(args);
    }
}
void ablate::monitors::logs::CsvLog::Print(const char *name, std::size_t num, const double *values, const char *formatIn) {
    if (file) {
        // set a default format if not specified
        const char *format = formatIn ? formatIn : "%g";

        for (std::size_t i = 0; i < num; i++) {
            PetscFPrintf(PETSC_COMM_SELF, file, format, values[i]) >> checkError;
            PetscFPrintf(PETSC_COMM_SELF, file, separator) >> checkError;
        }
    }
}

#include "parser/registrar.hpp"
REGISTER(ablate::monitors::logs::Log, ablate::monitors::logs::CsvLog, "Writes the result of the log to a csv file.  Only prints data to the log.",
         ARG(std::string, "name", "the name of the log file"));
