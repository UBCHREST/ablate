#ifndef ABLATELIBRARY_CONSTANTVALUE_HPP
#define ABLATELIBRARY_CONSTANTVALUE_HPP

#include "mathFunction.hpp"
#include "memory"

namespace ablate::mathFunctions {

class ConstantValue : public MathFunction {
   private:
    const std::vector<double> value;
    const bool uniformValue;
    static PetscErrorCode ConstantValuePetscFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);
    static PetscErrorCode ConstantValueUniformPetscFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

   public:
    explicit ConstantValue(double value);

    explicit ConstantValue(std::vector<double> values);

    explicit ConstantValue(std::initializer_list<double> values);

    [[nodiscard]] double Eval(const double& x, const double& y, const double& z, const double& t) const override;

    double Eval(const double* xyz, const int& ndims, const double& t) const override;

    void Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const override;

    void Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const override;

    PetscFunction GetPetscFunction() override { return uniformValue ? ConstantValueUniformPetscFunction : ConstantValuePetscFunction; }

    void* GetContext() override { return (void*)value.data(); }
};

}  // namespace ablate::mathFunctions
#endif  // ABLATELIBRARY_CONSTANTVALUE_HPP
