#ifndef ABLATELIBRARY_MERGELABELS_HPP
#define ABLATELIBRARY_MERGELABELS_HPP

#include <domain/region.hpp>
#include "modifier.hpp"

namespace ablate::domain::modifiers {
/**
 * Merge the specified labels/values into a combined new region
 */
class MergeLabels :public Modifier {
   private:
    const std::shared_ptr<domain::Region> mergedRegion;
    const std::vector<std::shared_ptr<domain::Region>> regions;

   public:
    MergeLabels(std::shared_ptr<domain::Region> mergedRegion, std::vector<std::shared_ptr<domain::Region>> regions);

    void Modify(DM&) override;

};

}

#endif  // ABLATELIBRARY_MERGELABELS_HPP
