#ifndef ABLATELIBRARY_FUNCTIONWRAPPER_HPP
#define ABLATELIBRARY_FUNCTIONWRAPPER_HPP
#include <functional>
#include "mathFunction.hpp"
namespace ablate::mathFunctions {

class FunctionWrapper : public MathFunction {
   public:
    typedef std::function<int(int dim, double time, const double x[], int nf, double* u, void* ctx)> Function;

   private:
    Function function;

    static PetscErrorCode WrappedPetscFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

   public:
    FunctionWrapper(const FunctionWrapper&) = delete;
    void operator=(const FunctionWrapper&) = delete;

    explicit FunctionWrapper(Function);

    double Eval(const double& x, const double& y, const double& z, const double& t) const override;

    double Eval(const double* xyz, const int& ndims, const double& t) const override;

    void Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const override;

    void Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const override;

    PetscFunction GetPetscFunction() override { return WrappedPetscFunction; }

    void* GetContext() override { return &function; }
};
}  // namespace ablate::mathFunctions
#endif  // ABLATELIBRARY_FUNCTIONWRAPPER_HPP
