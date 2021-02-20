#ifndef ABLATELIBRARY_ARGUMENTIDENTIFIER_HPP
#define ABLATELIBRARY_ARGUMENTIDENTIFIER_HPP

#define ARG(interfaceTypeFullName, inputName, description) ablate::parser::ArgumentIdentifier<interfaceTypeFullName>{inputName, description}

namespace ablate::parser {
template <typename Interface>
struct ArgumentIdentifier {
    const std::string inputName;
    const std::string description;
};
}  // namespace ablate::parser

#endif  // ABLATELIBRARY_ARGUMENTIDENTIFIER_HPP
