#ifndef ABLATELIBRARY_CUTLABEL_HPP
#define ABLATELIBRARY_CUTLABEL_HPP

#include <domain/region.hpp>
#include "modifier.hpp"

namespace ablate::domain::modifiers {
/**
 * Remove the specified regions from the cutRegion
 */
class CutLabel : public Modifier {
   private:
    const std::shared_ptr<domain::Region> cutRegion;
    const std::vector<std::shared_ptr<domain::Region>> regions;

   public:
    CutLabel(std::shared_ptr<domain::Region> cutRegion, std::vector<std::shared_ptr<domain::Region>> regions);

    void Modify(DM&) override;
};

}  // namespace ablate::domain::modifiers

#endif  // ABLATELIBRARY_MERGELABELS_HPP
