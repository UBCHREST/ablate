#ifndef ABLATELIBRARY_FUNCTIONFACTORY_HPP
#define ABLATELIBRARY_FUNCTIONFACTORY_HPP
#include <memory>
#include <type_traits>
#include "functionPointer.hpp"
#include "functionWrapper.hpp"
#include "mathFunction.hpp"
#include "parsedFunction.hpp"

namespace ablate::mathFunctions {

std::shared_ptr<MathFunction> Create(std::string func) { return std::make_shared<ParsedFunction>(func); }

std::shared_ptr<MathFunction> Create(char* func) { return std::make_shared<ParsedFunction>(func); }

std::shared_ptr<MathFunction> Create(const char* func) { return std::make_shared<ParsedFunction>(func); }

template <typename F>
std::shared_ptr<MathFunction> Create(F func) {
    if constexpr (std::is_convertible<F, PetscFunction>::value) {
        return std::make_shared<FunctionPointer>(func);
    }
    return std::make_shared<FunctionWrapper>(func);
}
}  // namespace ablate::mathFunctions
#endif  // ABLATELIBRARY_FUNCTIONFACTORY_HPP
