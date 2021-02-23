#ifndef ABLATELIBRARY_PARSEDFUNCTION_HPP
#define ABLATELIBRARY_PARSEDFUNCTION_HPP
#include <muParser.h>
#include "mathFunction.hpp"

namespace ablate::mathFunctions{
/**
 * simple wrapper to compute a function from a x,y,z string.
 * see https://beltoforion.de/en/muparser/index.php
 */

class ParsedFunction : public MathFunction {
   private:
    mutable double coordinate[3] = {0, 0, 0};
    mutable double time = 0.0;
    mu::Parser parser;
    const std::string formula;

   private:
    static PetscErrorCode ParsedPetscFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);

   public:
    ParsedFunction(const ParsedFunction&) = delete;
    void operator=(const ParsedFunction&) = delete;

    explicit ParsedFunction(std::string functionString);

    double Eval(const double& x, const double& y, const double& z, const double &t) const override;

    double Eval(const double* xyz, const int& ndims, const double &t) const override;

    void Eval(const double& x, const double& y, const double& z, const double &t, std::vector<double>& result) const override;

    void Eval(const double* xyz, const int& ndims, const double &t, std::vector<double>& result) const override;

    void* GetContext() override{
        return this;
    }

    PetscFunction GetPetscFunction() override{
        return ParsedPetscFunction;
    }
};
}

#endif  // ABLATELIBRARY_PARSEDFUNCTION_HPP
