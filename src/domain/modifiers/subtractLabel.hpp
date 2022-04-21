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
    const std::shared_ptr<domain::Region> subtrahendRegion;

   public:
    SubtractLabel(std::shared_ptr<domain::Region> differenceRegion, std::shared_ptr<domain::Region> minuendRegion, std::shared_ptr<domain::Region> subtrahendRegion);

    void Modify(DM&) override;

    std::string ToString() const override;
};

}  // namespace ablate::domain::modifiers

#endif  // ABLATELIBRARY_MERGELABELS_HPP
