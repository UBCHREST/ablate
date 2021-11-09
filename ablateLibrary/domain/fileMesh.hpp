#ifndef ABLATELIBRARY_FILEMESH_HPP
#define ABLATELIBRARY_FILEMESH_HPP

#include <filesystem>
#include <parameters/parameters.hpp>
#include "domain.hpp"

namespace ablate::domain {

class FileMesh : public Domain {
   private:
    static DM ReadDMFromFile(const std::string& name, const std::filesystem::path& path);

   public:
    explicit FileMesh(std::string nameIn, std::filesystem::path path, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors, std::vector<std::shared_ptr<modifiers::Modifier>> modifiers = {});
    ~FileMesh() override;
};
};      // namespace ablate::domain
#endif  // ABLATELIBRARY_FILEMESH_HPP
