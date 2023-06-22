#ifndef ABLATELIBRARY_FIELDFUNCTION_HPP
#define ABLATELIBRARY_FIELDFUNCTION_HPP
#include <memory>
#include <string>
#include "domain/region.hpp"
#include "mathFunctions/mathFunction.hpp"

namespace ablate::mathFunctions {
/**
 * Class used to describe initial or exaction conditions of a field using a math function
 */
class FieldFunction {
   private:
    const std::shared_ptr<mathFunctions::MathFunction> solutionField;
    const std::shared_ptr<mathFunctions::MathFunction> timeDerivative;
    const std::string fieldName;
    const std::shared_ptr<ablate::domain::Region> region;

   public:
    /*
     * Public constructor for field functions that has both
     */
    FieldFunction(std::string fieldName, std::shared_ptr<mathFunctions::MathFunction> solutionField, std::shared_ptr<mathFunctions::MathFunction> timeDerivative = {},
                  std::shared_ptr<ablate::domain::Region> region = nullptr);

    /**
     * The name of the field
     * @return
     */
    [[nodiscard]] const std::string& GetName() const { return fieldName; }

    /**
     * Returns bool if this field function has a solution field
     * @return
     */
    [[nodiscard]] bool HasSolutionField() const { return solutionField != nullptr; }

    /**
     * Return reference to the solution field
     * @return
     */
    mathFunctions::MathFunction& GetSolutionField() { return *solutionField; }

    /**
     * Return pointer to math function for field
     * @return
     */
    [[nodiscard]] std::shared_ptr<mathFunctions::MathFunction> GetFieldFunction() const { return solutionField; }

    /**
     * Returns bool if this field function has a time derivative field
     * @return
     */
    [[nodiscard]] bool HasTimeDerivative() const { return timeDerivative != nullptr; }

    /**
     * Return reference to the time derivative field
     * @return
     */
    mathFunctions::MathFunction& GetTimeDerivative() { return *timeDerivative; }

    /**
     * Return pointer to time derivative field
     * @return
     */
    [[nodiscard]] std::shared_ptr<mathFunctions::MathFunction> GetTimeDerivativeFunction() const { return timeDerivative; }

    /**
     * Return the valid function for this field function
     * @return
     */
    [[nodiscard]] const std::shared_ptr<ablate::domain::Region>& GetRegion() const { return region; }
};
}  // namespace ablate::mathFunctions
#endif  // ABLATELIBRARY_FIELDFUNCTION_HPP
