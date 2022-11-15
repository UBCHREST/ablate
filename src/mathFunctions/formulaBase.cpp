#include "formulaBase.hpp"

#include <tgmath.h>
#include <utility>
#include "utilities/stringUtilities.hpp"

ablate::mathFunctions::FormulaBase::FormulaBase(std::string functionString, const std::shared_ptr<ablate::parameters::Parameters>& constants) : formula(std::move(functionString)) {
    // define the x,y,z and t variables
    parser.DefineVar("x", &coordinate[0]);
    parser.DefineVar("y", &coordinate[1]);
    parser.DefineVar("z", &coordinate[2]);
    parser.DefineVar("t", &time);

    // Add in any provided constants
    if (constants) {
        for (const auto& key : constants->GetKeys()) {
            parser.DefineConst(key, constants->GetExpect<double>(key));
        }
    }

    // add in any additional helper functions
    if (ablate::utilities::StringUtilities::Contains(formula, "Power")) {
        parser.DefineFun("Power", PowerFunction, true);
    }
    // check for random number
    if (ablate::utilities::StringUtilities::Contains(formula, "pRand")) {
        parser.DefineFunUserData("pRand", PseudoRandomFunction, reinterpret_cast<void*>(&pseudoRandomEngine), false);
    }
    if (ablate::utilities::StringUtilities::Contains(formula, "rand")) {
        std::random_device rd;
        randomEngine = std::default_random_engine(rd());
        parser.DefineFunUserData("rand", RandomFunction, reinterpret_cast<void*>(&randomEngine), false);
    }
    if (ablate::utilities::StringUtilities::Contains(formula, "%")) {
        parser.DefineOprt("%", ModulusOperator, 0, mu::oaLEFT, true);
    }

    // set the expression
    parser.SetExpr(formula);
}

std::invalid_argument ablate::mathFunctions::FormulaBase::ConvertToException(mu::Parser::exception_type& exception) {
    return std::invalid_argument("Unable to parser (" + exception.GetExpr() + "). " + exception.GetMsg());
}
mu::value_type ablate::mathFunctions::FormulaBase::PowerFunction(mu::value_type a, mu::value_type b) { return PetscPowReal(a, b); }

mu::value_type ablate::mathFunctions::FormulaBase::RandomFunction(void* data, mu::value_type lowerBound, mu::value_type upperBound) {
    std::uniform_real_distribution<mu::value_type> uniformDist(lowerBound, upperBound);
    return uniformDist(*reinterpret_cast<std::default_random_engine*>(data));
}

mu::value_type ablate::mathFunctions::FormulaBase::PseudoRandomFunction(void* data, mu::value_type lowerBound, mu::value_type upperBound) {
    std::uniform_real_distribution<mu::value_type> uniformDist(lowerBound, upperBound);
    return uniformDist(*reinterpret_cast<std::minstd_rand0*>(data));
}
mu::value_type ablate::mathFunctions::FormulaBase::ModulusOperator(mu::value_type left, mu::value_type right) { return std::fmod(left, right); }
