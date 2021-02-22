#ifndef ABLATELIBRARY_ARGUMENTIDENTIFIER_HPP
#define ABLATELIBRARY_ARGUMENTIDENTIFIER_HPP
#include <string>

#define TMP_COMMA ,

#define ARG(interfaceTypeFullName, inputName, description) ablate::parser::ArgumentIdentifier<interfaceTypeFullName>{inputName, description}

namespace ablate::parser {
template <typename Interface>
struct ArgumentIdentifier {
    const std::string inputName;
    const std::string description;

    bool operator==(const ArgumentIdentifier<Interface>& other) const {
        return inputName == other.inputName && description == other.description;
    }
};
}  // namespace ablate::parser

#endif  // ABLATELIBRARY_ARGUMENTIDENTIFIER_HPP
