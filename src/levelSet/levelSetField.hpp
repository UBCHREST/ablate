#ifndef ABLATELIBRARY_LEVELSETFIELDS_HPP
#define ABLATELIBRARY_LEVELSETFIELDS_HPP

#include <domain/region.hpp>
#include <memory>
#include <string>
#include <vector>
#include "domain/fieldDescriptor.hpp"
#include "parameters/mapParameters.hpp"
#include "domain/fieldDescription.hpp"


namespace ablate::levelSet {

class LevelSetField : public domain::FieldDescriptor {
   private:
    const std::shared_ptr<domain::Region> region;

   public:
    LevelSetField(std::shared_ptr<domain::Region> = {});

    std::vector<std::shared_ptr<domain::FieldDescription>> GetFields() override;
};

}  // namespace ablate::levelSet

#endif  // ABLATELIBRARY_LEVELSETFIELDS_HPP




