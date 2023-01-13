#ifndef ABLATELIBRARY_SIMPLEFORMULA_HPP
#define ABLATELIBRARY_SIMPLEFORMULA_HPP
#include <muParser.h>
#include "formulaBase.hpp"

namespace ablate::mathFunctions {
/**
 * simple wrapper to compute a function from a x,y,z string.
 * see https://beltoforion.de/en/muparser/index.php
 */

class SimpleFormula : public FormulaBase {
   private:
    static PetscErrorCode ParsedPetscFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

   public:
    SimpleFormula(const SimpleFormula&) = delete;
    void operator=(const SimpleFormula&) = delete;

    explicit SimpleFormula(std::string functionString);

    double Eval(const double& x, const double& y, const double& z, const double& t) const override;

    double Eval(const double* xyz, const int& ndims, const double& t) const override;

    void Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const override;

    void Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const override;

    void* GetContext() override { return this; }

    PetscFunction GetPetscFunction() override { return ParsedPetscFunction; }
};
}  // namespace ablate::mathFunctions

#endif  // ABLATELIBRARY_SIMPLEFORMULA_HPP
