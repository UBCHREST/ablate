#include "log.hpp"

ablate::monitors::logs::Log::~Log() {
    if (ostream) {
        ostream->flush();
    }
}

void ablate::monitors::logs::Log::Print(const char* name, std::size_t num, const double* values, const char* formatIn) {
    // print the name
    Printf("%s: ", name);

    // set a default format if not specified
    const char* format = formatIn ? formatIn : "%g";

    // Print the start of the array
    Print("[");

    // print the first value
    if (num > 0) {
        Printf(format, values[0]);
    }

    // Now the rest of the arrays
    for (std::size_t c = 1; c < num; c++) {
        Print(", ");
        Printf(format, values[c]);
    }
    // close the array
    Print("]");
}

void ablate::monitors::logs::Log::Print(const char* name, const std::vector<double>& values, const char* format) { Print(name, values.size(), &values[0], format); }

std::ostream& ablate::monitors::logs::Log::GetStream() {
    if (!ostream) {
        ostreambuf = std::make_unique<DefaultOutBuffer>(*this);
        ostream = std::make_unique<std::ostream>(ostreambuf.get());
    }
    return *ostream;
}
