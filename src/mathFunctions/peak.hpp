#ifndef ABLATELIBRARY_PEAK_FUNTION_HPP
#define ABLATELIBRARY_PEAK_FUNTION_HPP
#include <muParser.h>
#include "formulaBase.hpp"

namespace ablate::mathFunctions {
/**
 * Peak functions in x, y, or z
 */

class Peak : public MathFunction {
   private:
    const std::vector<double> startValue;
    const std::vector<double> peakValue;
    const std::vector<double> endValue;
    const double start;
    const double peak;
    const double end;
    const int dir;

    static PetscErrorCode PeakPetscFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

    /**
     *
     * @param x
     * @param xS the start location
     * @param xP the peak location
     * @param xE the end location
     * @param yS the start value
     * @param yP the peak value
     * @param yE the end value
     * @return
     */
    inline static double Interpolate(double x, double xS, double xP, double xE, double yS, double yP, double yE) {
        if (x < xS) {
            return yS;
        } else if (x > xE) {
            return yE;
        } else if (x < xP) {
            return yS + (x - xS) * (yP - yS) / (xP - xS);
        } else {
            return yP + (x - xP) * (yE - yP) / (xE - xP);
        }
    }

    /**
     * Helper function to determine direction
     * @param x
     * @param x0
     * @param x1
     * @param y0
     * @param y1
     * @return
     */
    inline double DetermineDirectionValue(double x, double y, double z) const {
        switch (dir) {
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            default:
                return x;
        }
    }

   public:
    Peak(const Peak&) = delete;
    void operator=(const Peak&) = delete;

    /**
     * Peak function from start to end with the specified start and end values
     * @param startValue
     * @param endValue
     * @param start
     * @param end
     * @param dir 0, 1, 2
     */
    explicit Peak(std::vector<double> startValue, std::vector<double> peakValue, std::vector<double> endValue, double start, double peak, double end, int dir);

    double Eval(const double& x, const double& y, const double& z, const double& t) const override;

    double Eval(const double* xyz, const int& ndims, const double& t) const override;

    void Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const override;

    void Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const override;

    void* GetContext() override { return this; }

    PetscFunction GetPetscFunction() override { return PeakPetscFunction; }
};
}  // namespace ablate::mathFunctions

#endif  // ABLATELIBRARY_SIMPLEFORMULA_HPP
