#ifndef ABLATELIBRARY_LOWMACHFLOWFIELDS_HPP
#define ABLATELIBRARY_LOWMACHFLOWFIELDS_HPP

#include <domain/region.hpp>
#include <memory>
#include <string>
#include <vector>
#include "domain/fieldDescriptor.hpp"
#include "eos/eos.hpp"

namespace ablate::finiteVolume {

class LowMachFlowFields : public domain::FieldDescriptor {
   private:
    const std::shared_ptr<domain::Region> region;
    const bool includeSourceTerms;

   public:
    LowMachFlowFields(std::shared_ptr<domain::Region> region = {}, bool includeSourceTerms = false);

    std::vector<std::shared_ptr<domain::FieldDescription>> GetFields() override;
};

}  // namespace ablate::finiteVolume

#endif  // ABLATELIBRARY_COMPRESSIBLEFLOWFIELDS_HPP
