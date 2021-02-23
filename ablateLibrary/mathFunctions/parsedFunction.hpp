#ifndef ABLATELIBRARY_PARSEDFUNCTION_HPP
#define ABLATELIBRARY_PARSEDFUNCTION_HPP
#include <muParser.h>
#include <petsc.h>

namespace ablate::mathFunctions{
/**
 * simple wrapper to compute a function from a x,y,z string.
 * see https://beltoforion.de/en/muparser/index.php
 */

typedef PetscErrorCode (*PetscFunction)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);

class ParsedFunction {
   private:
    mutable double coordinate[3] = {0, 0, 0};
    mutable double time = 0.0;
    mu::Parser parser;

   private:
    static PetscErrorCode ParsedPetscFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);

   public:
    ParsedFunction(const ParsedFunction&) = delete;
    void operator=(const ParsedFunction&) = delete;

    explicit ParsedFunction(std::string functionString);

    double Eval(const double& x, const double& y, const double& z, const double &t) const;

    double Eval(const double* xyz, const int& ndims, const double &t) const;

    void Eval(const double& x, const double& y, const double& z, const double &t, std::vector<double>& result) const;

    void Eval(const double* xyz, const int& ndims, const double &t, std::vector<double>& result) const;

    void* GetContext(){
        return this;
    }

    PetscFunction GetPetscFunction(){
        return ParsedPetscFunction;
    }

};
}

#endif  // ABLATELIBRARY_PARSEDFUNCTION_HPP
