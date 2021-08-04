#ifndef ABLATELIBRARY_LINEARINTERPOLATOR_HPP
#define ABLATELIBRARY_LINEARINTERPOLATOR_HPP

#include <filesystem>
#include <istream>
#include <vector>
#include "mathFunction.hpp"
namespace ablate::mathFunctions {

/**
 * a simple interpolator that reads a text file and interpolates the value.  The xAxisColumn is assumed to be monotonic.
 * An example input would look like
 * x,y,z
 * 1,2,3
 * 2,2,1
 */
class LinearInterpolator : public MathFunction {
   private:
    std::vector<double> xValues;
    std::vector<std::vector<double>> yValues;
    const std::string xColumn;
    const std::vector<std::string> yColumns;
    const std::shared_ptr<MathFunction> locationToXCoordFunction;

   private:
    void ParseInputData(std::istream& inputFile);

    void Interpolate(double x, size_t numInterpolations, double* result) const;

    static PetscErrorCode LinearInterpolatorPetscFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

   public:
    LinearInterpolator(std::filesystem::path inputFile, std::string xColumn, std::vector<std::string> yColumns, std::shared_ptr<MathFunction> locationToXCoord);
    LinearInterpolator(std::istream& inputFile, std::string xColumn, std::vector<std::string> yColumns, std::shared_ptr<MathFunction> locationToXCoord);

    double Eval(const double& x, const double& y, const double& z, const double& t) const override;

    double Eval(const double* xyz, const int& ndims, const double& t) const override;

    void Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const override;

    void Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const override;

    void* GetContext() override { return this; }

    PetscFunction GetPetscFunction() override { return LinearInterpolatorPetscFunction; }

    const std::vector<double>& GetXValues() const { return xValues; }

    const std::vector<std::vector<double>>& GetYValues() const { return yValues; }
};
}  // namespace ablate::mathFunctions
#endif  // ABLATELIBRARY_LINEARINTERPOLATOR_HPP
