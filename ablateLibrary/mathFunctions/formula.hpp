#ifndef ABLATELIBRARY_FORMULA_HPP
#define ABLATELIBRARY_FORMULA_HPP
#include <muParser.h>
#include <memory>
#include "parameters/parameters.hpp"
#include <vector>
#include "mathFunction.hpp"

namespace ablate::mathFunctions {
class Formula : public MathFunction {
   private:
    mutable double coordinate[3] = {0, 0, 0};
    mutable double time = 0.0;

    mu::Parser parser;
    const std::string formula;

    // store the scratch variables
    std::vector<std::unique_ptr<double>> nestedValues;
    std::vector<std::shared_ptr<MathFunction>> nestedFunctions;

   private:
    static PetscErrorCode ParsedPetscNested(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

   public:
    Formula(const Formula&) = delete;
    void operator=(const Formula&) = delete;

    explicit Formula(std::string functionString, std::map<std::string, std::shared_ptr<MathFunction>>, std::shared_ptr<ablate::parameters::Parameters> constants = {});

    double Eval(const double& x, const double& y, const double& z, const double& t) const override;

    double Eval(const double* xyz, const int& ndims, const double& t) const override;

    void Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const override;

    void Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const override;

    void* GetContext() override { return this; }

    PetscFunction GetPetscFunction() override { return ParsedPetscNested; }
};
}  // namespace ablate::mathFunctions
#endif  // ABLATELIBRARY_FORMULA_HPP
