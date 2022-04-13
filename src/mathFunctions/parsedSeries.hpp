#ifndef ABLATELIBRARY_PARSEDSERIES_HPP
#define ABLATELIBRARY_PARSEDSERIES_HPP
#include <muParser.h>
#include "mathFunction.hpp"

namespace ablate::mathFunctions {
/**
 * computes a series result from a string function with variables x, y, z, t, and n where i index of summation. See the ParsedFunction for details on the string formatting.
 * Note that the lower and upper bound are inclusive.
 */

class ParsedSeries : public MathFunction {
   private:
    mutable double coordinate[3] = {0, 0, 0};
    mutable double time = 0.0;
    mutable double i = 0;

    mu::Parser parser;
    const std::string formula;
    const int lowerBound;
    const int upperBound;

   private:
    static PetscErrorCode ParsedPetscSeries(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

   public:
    ParsedSeries(const ParsedSeries&) = delete;
    void operator=(const ParsedSeries&) = delete;

    explicit ParsedSeries(std::string functionString, int lowerBound = 1, int upperBound = 1000);

    double Eval(const double& x, const double& y, const double& z, const double& t) const override;

    double Eval(const double* xyz, const int& ndims, const double& t) const override;

    void Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const override;

    void Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const override;

    void* GetContext() override { return this; }

    PetscFunction GetPetscFunction() override { return ParsedPetscSeries; }
};
}  // namespace ablate::mathFunctions

#endif  // ABLATELIBRARY_PARSEDFUNCTION_HPP
