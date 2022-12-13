#ifndef ABLATELIBRARY_EQUATIONINTERVAL_HPP
#define ABLATELIBRARY_EQUATIONINTERVAL_HPP

#include <muParser.h>
#include "interval.hpp"
#include "mathFunctions/formulaBase.hpp"

namespace ablate::io::interval {

/**
 * Determines if the interval is valid based upon a user supplied equation.
 * If the result of the equation is > 0, then check will be true, else it will be false.  The custom variables
 * for this formula are "time" and "step".  The class privately extends FormulaBase to allow use of the
 * prebuilt custom functions.
 */
class EquationInterval : public Interval, private ablate::mathFunctions::FormulaBase {
   private:
    //! the steps linked to the parser
    mu::value_type step;

   public:
    /**
     * If the result of the equation is > 0, then check will be true.  Else it will be false.  The custom variables
     * for this formula are "time" and "step"
     * @param functionString
     */
    explicit EquationInterval(std::string functionString);

    /**
     * uses the supplied equation to determine check
     * @param comm
     * @param steps
     * @param time
     * @return
     */
    bool Check(MPI_Comm comm, PetscInt step, PetscReal time) override;

   private:
    //! empty method to allow extending ablate::mathFunctions::FormulaBase
    double Eval(const double& x, const double& y, const double& z, const double& t) const override { return NAN; }

    //! empty method to allow extending ablate::mathFunctions::FormulaBase
    double Eval(const double* xyz, const int& ndims, const double& t) const override { return NAN; }

    //! empty method to allow extending ablate::mathFunctions::FormulaBase
    void Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const override {}

    //! empty method to allow extending ablate::mathFunctions::FormulaBase
    void Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const override {}

    //! empty method to allow extending ablate::mathFunctions::FormulaBase
    void* GetContext() override { return nullptr; }

    //! empty method to allow extending ablate::mathFunctions::FormulaBase
    mathFunctions::PetscFunction GetPetscFunction() override { return nullptr; }
};

}  // namespace ablate::io::interval
#endif
