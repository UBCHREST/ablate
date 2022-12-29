#include "mpiFileLog.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscUtilities.hpp"
#include "environment/runEnvironment.hpp"

ablate::monitors::logs::MpiFileLog::MpiFileLog(std::string fileName)
    : outputPath(std::filesystem::path(fileName).is_absolute() ? std::filesystem::path(fileName) : ablate::environment::RunEnvironment::Get().GetOutputDirectory() / fileName), file(nullptr) {}

ablate::monitors::logs::MpiFileLog::~MpiFileLog() {
    if (file) {
        fflush(file);
        fclose(file);
    }
}

void ablate::monitors::logs::MpiFileLog::Initialize(MPI_Comm commIn) {
    Log::Initialize(commIn);

    // Update the output path with the rank
    int rank;
    MPI_Comm_rank(commIn, &rank) >> utilities::MpiUtilities::checkError;
    outputPath = outputPath.parent_path() / (outputPath.stem().string() + "." + std::to_string(rank) + outputPath.extension().string());
    file = fopen(outputPath.c_str(), "a");
}

void ablate::monitors::logs::MpiFileLog::Printf(const char* format, ...) {
    if (file) {
        va_list args;
        va_start(args, format);
        PetscVFPrintf(file, format, args) >> utilities::PetscUtilities::checkError;
        va_end(args);
    }
}

#include "registrar.hpp"
REGISTER(ablate::monitors::logs::Log, ablate::monitors::logs::MpiFileLog, "Writes the log of each rank to a separate file such that (file.txt => file.0.txt).  ",
         ARG(std::string, "name", "the base name of the log file"));
