#ifndef ABLATELIBRARY_DEMANGLER_H
#define ABLATELIBRARY_DEMANGLER_H

#include <map>
#include <string>
#include <vector>
#include <filesystem>
namespace ablate::utilities {
class Demangler {
   public:
    static std::string Demangle(const std::string&);

   private:
    inline static std::map<std::string, std::string> prettyNames = {
        {typeid(std::string).name(), "string"},
        {typeid(std::map<std::string, std::string>).name(), "argument map"},
        {typeid(std::vector<int>).name(), "int list"},
        {typeid(std::vector<double>).name(), "double list"},
        {typeid(std::vector<std::string>).name(), "string list"},
        {typeid(std::filesystem::path).name(), "file path or url"},
    };
};
}  // namespace ablate::utilities

#endif  // ABLATELIBRARY_DEMANGLER_H
