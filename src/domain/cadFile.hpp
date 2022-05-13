#ifndef ABLATELIBRARY_CADFILE_HPP
#define ABLATELIBRARY_CADFILE_HPP

#include <filesystem>
#include <parameters/parameters.hpp>
#include "domain.hpp"

namespace ablate::domain {

class CadFile : public Domain {
   private:
    //! read in the cad file and expand to volumetric mesh
    static DM ReadDMFromCadFile(const std::string& name, const std::filesystem::path& path, const std::shared_ptr<parameters::Parameters>& surfaceOptions);

   public:
    explicit CadFile(const std::string& nameIn, const std::filesystem::path& path, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors,
                     std::vector<std::shared_ptr<modifiers::Modifier>> modifiers = {}, const std::shared_ptr<parameters::Parameters>& options = {},
                     const std::shared_ptr<parameters::Parameters>& surfaceOptions = {});

    ~CadFile() override;
};

}  // namespace ablate::domain
#endif  // ABLATELIBRARY_CADFILE_HPP
