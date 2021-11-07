#ifndef ABLATELIBRARY_ARGUMENTIDENTIFIER_HPP
#define ABLATELIBRARY_ARGUMENTIDENTIFIER_HPP
#include <optional>
#include <string>
#include "enumWrapper.hpp"

#define TMP_COMMA ,

#define ARG(interfaceTypeFullName, inputName, description) \
    ablate::parser::ArgumentIdentifier<interfaceTypeFullName> { inputName, description, false }

#define OPT(interfaceTypeFullName, inputName, description) \
    ablate::parser::ArgumentIdentifier<interfaceTypeFullName> { inputName, description, true }

#define ENUM(interfaceTypeFullName, inputName, description) \
    ablate::parser::ArgumentIdentifier<ablate::parser::EnumWrapper<interfaceTypeFullName>> { inputName, description, false }

namespace ablate::parser {
template <typename Interface>
struct ArgumentIdentifier {
    const std::string inputName;
    const std::string description;
    const bool optional;
    bool operator==(const ArgumentIdentifier<Interface>& other) const { return inputName == other.inputName && optional == other.optional; }
};

template <typename Interface>
std::ostream& operator<<(std::ostream& os, const ArgumentIdentifier<Interface>& arg) {
    os << arg.inputName << (arg.optional ? "(OPT)" : "") << ": " << arg.description;
    return os;
}

}  // namespace ablate::parser

#endif  // ABLATELIBRARY_ARGUMENTIDENTIFIER_HPP
