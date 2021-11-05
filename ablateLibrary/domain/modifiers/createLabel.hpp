#ifndef ABLATELIBRARY_CREATELABEL_HPP
#define ABLATELIBRARY_CREATELABEL_HPP

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
    const std::string name;

    // function to determine the label value
    std::shared_ptr<mathFunctions::MathFunction> function;

    // The depth to evaluate the label.
    const PetscInt dmHeight;

    // value to assign if the function evaluates positive
    const PetscInt labelValue;

   public:
    explicit CreateLabel(std::string name, std::shared_ptr<mathFunctions::MathFunction> function, int dmDepth = {}, int labelValue = {});

    void Modify(DM&) override;
};

}  // namespace ablate::domain::modifier
#endif  // ABLATELIBRARY_CREATELABEL_HPP
