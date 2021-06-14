#ifndef ABLATELIBRARY_FUNCTIONFACTORY_HPP
#define ABLATELIBRARY_FUNCTIONFACTORY_HPP
#include <memory>
#include <type_traits>
#include "constantValue.hpp"
#include "functionPointer.hpp"
#include "functionWrapper.hpp"
#include "mathFunction.hpp"
#include "parsedFunction.hpp"

namespace ablate::mathFunctions {

inline std::shared_ptr<MathFunction> Create(std::string func) { return std::make_shared<ParsedFunction>(func); }

inline std::shared_ptr<MathFunction> Create(char* func) { return std::make_shared<ParsedFunction>(func); }

inline std::shared_ptr<MathFunction> Create(const char* func) { return std::make_shared<ParsedFunction>(func); }

inline std::shared_ptr<MathFunction> Create(double value) { return std::make_shared<ConstantValue>(value); }

inline std::shared_ptr<MathFunction> Create(std::vector<double> value) { return std::make_shared<ConstantValue>(value); }

template <typename F>
inline std::shared_ptr<MathFunction> Create(F func) {
    if constexpr (std::is_convertible<F, PetscFunction>::value) {
        return std::make_shared<FunctionPointer>(func);
    }
    return std::make_shared<FunctionWrapper>(func);
}

template <typename F>
inline std::shared_ptr<MathFunction> Create(F func, void* context) {
    return std::make_shared<FunctionPointer>(func, context);
}
}  // namespace ablate::mathFunctions
#endif  // ABLATELIBRARY_FUNCTIONFACTORY_HPP
