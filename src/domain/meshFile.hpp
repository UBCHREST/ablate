#ifndef ABLATELIBRARY_MESHFILE_HPP
#define ABLATELIBRARY_MESHFILE_HPP

#include <filesystem>
#include <parameters/parameters.hpp>
#include "domain.hpp"

namespace ablate::domain {

class MeshFile : public Domain {
   private:
    static DM ReadDMFromFile(const std::string& name, const std::filesystem::path& path);

   public:
    explicit MeshFile(const std::string& nameIn, const std::filesystem::path& path, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors,
                      std::vector<std::shared_ptr<modifiers::Modifier>> modifiers = {}, const std::shared_ptr<parameters::Parameters>& options = {});
    ~MeshFile() override;
};
};      // namespace ablate::domain
#endif  // ABLATELIBRARY_FILEMESH_HPP
