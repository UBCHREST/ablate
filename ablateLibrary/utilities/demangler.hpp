#ifndef ABLATELIBRARY_DEMANGLER_H
#define ABLATELIBRARY_DEMANGLER_H

#include <map>
#include <string>
namespace ablate::utilities {
class Demangler {
   public:
    static std::string Demangle(const std::string&);

   private:
    inline static std::map<std::string, std::string> prettyNames = {
        {typeid(std::string).name(), "string"},
        {typeid(std::map<std::string, std::string>).name(), "argument map"},
    };
};
}  // namespace ablate::utilities

#endif  // ABLATELIBRARY_DEMANGLER_H
