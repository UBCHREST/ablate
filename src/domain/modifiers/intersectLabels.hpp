#ifndef ABLATELIBRARY_INTERSECTLABELS_HPP
#define ABLATELIBRARY_INTERSECTLABELS_HPP

#include <domain/region.hpp>
#include "modifier.hpp"

namespace ablate::domain::modifiers {
/**
 * Create a label of the intersection regions the specified labels/values into a combined new region
 */
class IntersectLabels : public Modifier {
   private:
    const std::shared_ptr<domain::Region> intersectRegion;
    const std::vector<std::shared_ptr<domain::Region>> regions;

   public:
    IntersectLabels(std::shared_ptr<domain::Region> intersectRegion, std::vector<std::shared_ptr<domain::Region>> regions);

    void Modify(DM&) override;

    std::string ToString() const override { return "ablate::domain::modifiers::IntersectLabels"; }
};

}  // namespace ablate::domain::modifiers

#endif  // ABLATELIBRARY_MERGELABELS_HPP
