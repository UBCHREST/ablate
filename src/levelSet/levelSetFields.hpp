#ifndef ABLATELIBRARY_LEVELSETFIELD_HPP
#define ABLATELIBRARY_LEVELSETFIELD_HPP

#include <domain/region.hpp>
#include <memory>
#include <string>
#include <cstring>
#include <vector>
#include "domain/fieldDescriptor.hpp"


namespace ablate::levelSet {

class LevelSetFields : public domain::FieldDescriptor {
   private:
    const std::shared_ptr<domain::Region> region;
    std::string shape;

   public:
    LevelSetFields(std::shared_ptr<domain::Region> region = {}, std::string shape = "");

    // Names of the fieds
    inline const static std::string LEVELSET_FIELD = "levelSet";
    inline const static std::string CURVATURE_FIELD = "curvature";
    inline const static std::string NORMAL_FIELD = "normal";

    std::vector<std::shared_ptr<domain::FieldDescription>> GetFields() override;

};

}  // namespace ablate::levelSet

#endif  // ABLATELIBRARY_LEVELSETFIELD_HPP




