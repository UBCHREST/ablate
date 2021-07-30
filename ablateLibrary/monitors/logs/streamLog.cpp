#include "streamLog.hpp"
#include <utilities/mpiError.hpp>
ablate::monitors::logs::StreamLog::StreamLog(std::ostream& stream) : stream(stream) {}

void ablate::monitors::logs::StreamLog::Initialize(MPI_Comm comm) {}

void ablate::monitors::logs::StreamLog::Print(const char* value) { stream << value; }

void ablate::monitors::logs::StreamLog::Printf(const char* format, ...) {
    va_list args;
    va_start(args, format);
    va_list argsCopy;
    va_copy(argsCopy, args);

    // try to print to the buffer
    auto reqSize = vsnprintf(&buffer[0], buffer.size(), format, args);

    if (reqSize > (int)buffer.size()) {
        buffer.resize(reqSize + 1);
        vsnprintf(&buffer[0], buffer.size(), format, argsCopy);
    }

    stream << &buffer[0];

    va_end(args);
    va_end(argsCopy);
}

#include "parser/registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::monitors::logs::Log, ablate::monitors::logs::StreamLog, "Writes to the std::cout stream");
