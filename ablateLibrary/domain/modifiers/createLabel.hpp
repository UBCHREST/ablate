#ifndef ABLATELIBRARY_CREATELABEL_HPP
#define ABLATELIBRARY_CREATELABEL_HPP

#include <domain/region.hpp>
#include <memory>
#include "mathFunctions/mathFunction.hpp"
#include "modifier.hpp"

namespace ablate::domain::modifiers {

/**
 * Class to create a label based upon a field function.  By default, positive values are assigned to the label while negative values are not.
 */
class CreateLabel : public Modifier {
   private:
    // label to create
    const std::shared_ptr<domain::Region> region;

    // function to determine the label value
    std::shared_ptr<mathFunctions::MathFunction> function;

    // The depth to evaluate the label.
    const PetscInt dmHeight;

   public:
    explicit CreateLabel(std::shared_ptr<domain::Region>, std::shared_ptr<mathFunctions::MathFunction> function, int dmDepth = {});

    void Modify(DM&) override;
};

}  // namespace ablate::domain::modifiers
#endif  // ABLATELIBRARY_CREATELABEL_HPP
