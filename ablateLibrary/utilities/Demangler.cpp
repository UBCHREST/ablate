#include "demangler.hpp"
#ifdef __GNUG__
#include <cxxabi.h>
#include <cstdlib>
#include <memory>

std::string ablate::utilities::Demangler::Demangle(const std::string& name) {
    // hard code some default values
    if (prettyNames.contains(name)) {
        return prettyNames[name];
    }

    int status = -4;  // some arbitrary value to eliminate the compiler warning

    // enable c++11 by passing the flag -std=c++11 to g++
    std::unique_ptr<char, void (*)(void*)> res{abi::__cxa_demangle(name.c_str(), NULL, NULL, &status), std::free};

    return (status == 0) ? res.get() : name;
}

#else

// does nothing if not g++
std::string ablate::utilities::Demangle(const std::string& name) { return name; }

#endif