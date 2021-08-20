#ifndef ABLATELIBRARY_PARSEDNESTED_HPP
#define ABLATELIBRARY_PARSEDNESTED_HPP

#include <muParser.h>
#include <memory>
#include <vector>
#include "mathFunction.hpp"

namespace ablate::mathFunctions {
class ParsedNested : public MathFunction {
   private:
    mutable double coordinate[3] = {0, 0, 0};
    mutable double time = 0.0;

    mu::Parser parser;
    const std::string formula;

    // store the scratch variables
    std::vector<double*> nestedValues;
    std::vector<std::shared_ptr<MathFunction>> nestedFunctions;

   private:
    static PetscErrorCode ParsedPetscNested(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

   public:
    ParsedNested(const ParsedNested&) = delete;
    void operator=(const ParsedNested&) = delete;

    explicit ParsedNested(std::string functionString, std::map<std::string, std::shared_ptr<MathFunction>>);
    ~ParsedNested() override;

    double Eval(const double& x, const double& y, const double& z, const double& t) const override;

    double Eval(const double* xyz, const int& ndims, const double& t) const override;

    void Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const override;

    void Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const override;

    void* GetContext() override { return this; }

    PetscFunction GetPetscFunction() override { return ParsedPetscNested; }
};
}  // namespace ablate::mathFunctions
#endif  // ABLATELIBRARY_PARSEDNESTED_HPP
