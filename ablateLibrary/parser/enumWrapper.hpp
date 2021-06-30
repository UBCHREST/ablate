#ifndef ABLATELIBRARY_ENUMWRAPPER_HPP
#define ABLATELIBRARY_ENUMWRAPPER_HPP
#include <optional>
#include <sstream>
#include <string>

namespace ablate::parser {
template <typename Enum>
class EnumWrapper {
   private:
    Enum value;

   public:
    EnumWrapper(Enum value) : value(value) {}

    EnumWrapper(std::string value) {
        std::istringstream stream(value);
        stream >> this->value;
    }

    operator Enum() const { return value; }
};
}  // namespace ablate::parser

#endif  // ABLATELIBRARY_ENUMWRAPPER_HPP
