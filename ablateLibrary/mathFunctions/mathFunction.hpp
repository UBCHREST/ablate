#ifndef ABLATELIBRARY_MATHFUNCTION_HPP
#define ABLATELIBRARY_MATHFUNCTION_HPP
#include <petsc.h>
#include <vector>

namespace ablate::mathFunctions {

typedef PetscErrorCode (*PetscFunction)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);

class MathFunction {
   public:
    virtual ~MathFunction() = default;

   public:
    virtual double Eval(const double& x, const double& y, const double& z, const double &t) const = 0;

    virtual double Eval(const double* xyz, const int& ndims, const double &t) const = 0;

    virtual void Eval(const double& x, const double& y, const double& z, const double &t, std::vector<double>& result) const =0;

    virtual void Eval(const double* xyz, const int& ndims, const double &t, std::vector<double>& result) const =0 ;

    virtual void* GetContext() =0 ;

    virtual PetscFunction GetPetscFunction() = 0;
};
}
#endif  // ABLATELIBRARY_MATHFUNCTION_HPP
