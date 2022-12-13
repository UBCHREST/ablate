#ifndef ABLATELIBRARY_FORMULABASE_HPP
#define ABLATELIBRARY_FORMULABASE_HPP

#include <muParser.h>
#include <random>
#include "mathFunction.hpp"
#include "parameters/parameters.hpp"

namespace ablate::mathFunctions {

/**
 * Formula base is the base abstract class shared by other formulas
 */
class FormulaBase : public MathFunction {
   private:
    //! Hold a random number engine always using the same seed
    std::minstd_rand0 pseudoRandomEngine{0};

    //! Hold a "real" random number engine
    std::default_random_engine randomEngine{0};

   protected:
    //! The coordinate linked to the parser
    mutable double coordinate[3] = {0, 0, 0};

    //! the time linked to the parser
    mutable double time = 0.0;

    //! The parser object library for this formula
    mu::Parser parser;

    //! the formula output for debugging
    const std::string formula;

    /**
     * protected constructor to build the formula base
     * @param functionString
     * @param constants
     */
    explicit FormulaBase(std::string functionString, const std::shared_ptr<ablate::parameters::Parameters>& constants);

    /**
     * helper function to convert to a invalid_exception
     * @param exception
     * @return
     */
    static std::invalid_argument ConvertToException(mu::Parser::exception_type& exception);

   public:
    //! prevent copy of this object
    FormulaBase(const FormulaBase&) = delete;
    //! prevent copy of this object
    void operator=(const FormulaBase&) = delete;

   private:
    /**
     * mu parser function to compute power given a^2
     * @param a
     * @param b
     * @return
     */
    static mu::value_type PowerFunction(mu::value_type a, mu::value_type b);

    /**
     * mu parser function to compute random number
     * @param lowerBound
     * @param upperBound
     * @return
     */
    static mu::value_type RandomFunction(void* data, mu::value_type lowerBound, mu::value_type upperBound);

    /**
     * mu parser function to compute deterministic pseudo random number
     * @param lowerBound
     * @param upperBound
     * @return
     */
    static mu::value_type PseudoRandomFunction(void* data, mu::value_type lowerBound, mu::value_type upperBound);

    /**
     * mu parser function to the modulus as left % right
     * @param lowerBound
     * @param upperBound
     * @return
     */
    static mu::value_type ModulusOperator(mu::value_type left, mu::value_type right);
};

}  // namespace ablate::mathFunctions
#endif  // ABLATELIBRARY_FORMULABASE_HPP
