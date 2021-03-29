#ifndef ABLATELIBRARY_FUNCTIONPOINTER_HPP
#define ABLATELIBRARY_FUNCTIONPOINTER_HPP
#include "mathFunction.hpp"

namespace ablate::mathFunctions {

class FunctionPointer : public MathFunction {
   private:
    void* context;
    PetscFunction function;

   public:
    explicit FunctionPointer(ablate::mathFunctions::PetscFunction function, void* context = nullptr);

    explicit FunctionPointer();

    double Eval(const double& x, const double& y, const double& z, const double& t) const override;

    double Eval(const double* xyz, const int& ndims, const double& t) const override;

    void Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const override;

    void Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const override;

    inline ablate::mathFunctions::PetscFunction GetPetscFunction() override { return function; }

    inline void* GetContext() override { return context; }
};

}  // namespace ablate::mathFunctions
#endif  // ABLATELIBRARY_FUNCTIONPOINTER_HPP
