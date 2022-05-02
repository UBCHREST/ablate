#ifndef ABLATELIBRARY_SUBTRACTLABEL_HPP
#define ABLATELIBRARY_SUBTRACTLABEL_HPP

#include <domain/region.hpp>
#include "modifier.hpp"

namespace ablate::domain::modifiers {
/**
 * Remove the specified regions from the region
 */
class SubtractLabel : public Modifier {
   private:
    const std::shared_ptr<domain::Region> differenceRegion;
    const std::shared_ptr<domain::Region> minuendRegion;
    const std::vector<std::shared_ptr<domain::Region>> subtrahendRegions;
    /**
     * determine if DMPlexLabelComplete should be called on each label
     * # External resources
     * [DMPlexLabelComplete Documentation](https://petsc.org/main/docs/manualpages/DMPLEX/DMPlexLabelComplete.html)
     */
    const bool incompleteLabel;

   public:
    SubtractLabel(std::shared_ptr<domain::Region> differenceRegion, std::shared_ptr<domain::Region> minuendRegion, std::vector<std::shared_ptr<domain::Region>> subtrahendRegions,
                  bool incompleteLabel = false);

    void Modify(DM&) override;

    std::string ToString() const override;
};

}  // namespace ablate::domain::modifiers

#endif  // ABLATELIBRARY_MERGELABELS_HPP
