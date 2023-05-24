#ifndef ABLATELIBRARY_MATHFUNCTION_HPP
#define ABLATELIBRARY_MATHFUNCTION_HPP
#include <petsc.h>
#include <vector>

namespace ablate::mathFunctions {

typedef PetscErrorCode (*PetscFunction)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

/**
 * Shared interface for all "math" style functions
 */
class MathFunction {
   public:
    /**
     * Return a single double value
     * @param x
     * @param y
     * @param z
     * @param t
     * @return
     */
    [[nodiscard]] virtual double Eval(const double& x, const double& y, const double& z, const double& t) const = 0;

    /**
     * Return a single double value based upon an xyz array
     * @param xyz
     * @param ndims
     * @param t
     * @return
     */
    [[nodiscard]] virtual double Eval(const double* xyz, const int& ndims, const double& t) const = 0;

    /**
     * Populate a result array
     * @param x
     * @param y
     * @param z
     * @param t
     * @param result
     */
    virtual void Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const = 0;

    /**
     * Populate a result array based upon an xyz array
     * @param xyz
     * @param ndims
     * @param t
     * @param result
     */
    virtual void Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const = 0;

    /**
     * Return a raw petsc style function to evaluate this math function
     * @return
     */
    virtual PetscFunction GetPetscFunction() = 0;

    /**
     * Return a context for petsc style functions
     * @return
     */
    virtual void* GetContext() = 0;

    /**
     * provide hook to allow math functions to cleanup
     */
    virtual ~MathFunction() = default;
};
}  // namespace ablate::mathFunctions
#endif  // ABLATELIBRARY_MATHFUNCTION_HPP
